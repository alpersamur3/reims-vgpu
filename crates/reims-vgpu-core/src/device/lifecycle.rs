//! The device's lifecycle: pause, unpause, stop and reset; suspend (`will_suspend`,
//! `finish_suspend`) and resume (`will_resume`, `did_resume`).
//!
//! ## Unlocked reads
//!
//! These paths read the root FIFO, the child FIFO slots, the guest displays, the root node, the
//! root page and the EFI registers without the device's locks, through the staged views
//! (`StagedMutex::atomics`): a thread holding the root FIFO's or the displays' lock (a stamp
//! wait report mid-describe, a page 0 or a detach) never blocks a suspend, a pause or an unpause
//! (`tests::unlocked_reads::*`,
//! `tests::guest_displays_paused::guest_displays_follow_while_display_mutex_is_held`). The root
//! FIFO's other registers have no unlocked view: they are read or stored under the root FIFO's lock
//! for that access alone, and every call out is made with the lock released. The child slots sit
//! behind a leaf lock of their own.
//!
//! ## Order
//!
//! - `pause` quiesces and drains the root FIFO, then the children, pauses the guest displays, and
//!   sets the paused state last: what the drains complete is published, with its interrupts, while
//!   the device still runs (`tests::pause::pauses`).
//! - `finish_suspend` drains and drops the children, drains the root FIFO, flushes and tears down
//!   every task under the task lock (what they hold and autorelease goes when the walk's holder
//!   and pool end, before the table empties:
//!   `tests::finish_suspend::the_walk_s_pool_pops_before_the_table_empties`), serializes, drops
//!   the root FIFO, then pauses the guest displays.
//! - `will_resume` stores `LeavingSuspendEarly` before it checks or reads the state
//!   (`tests::will_resume::nil_state`).
//! - `did_resume` restores the EFI display, the root page, the root FIFO, the children, the tasks
//!   and the display pages, then runs, finishes the guest displays' restore, starts the FIFOs and
//!   raises one interrupt (`tests::did_resume::nothing_to_rebuild`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::device::{DeviceCore, DeviceDeath, EfiRegisters, RunState};
use crate::display::guest_display::GuestDisplay;
use crate::fifo::Fifo;
use crate::guest::GuestPtr;
use crate::guest::abi::device_info::{check_capabilities, write_device_info};
use crate::guest::abi::suspend_blob::RESUME_ERROR_INVALID_CONTENT;
use crate::guest::abi::suspend_blob::{
    KeyValuePair, SERIALIZED_INFO_PAIRS, SuspendDeviceState, SuspendState,
};
use crate::guest::records::{APV_PAGE_SHIFT, APV_PAGE_SIZE};
use crate::num::{ToUsize, Truncate};
use crate::platform::Platform;
use crate::platform::device::GpuDevice as _;
use crate::raise::{OnFault, RaiseResult};
use crate::rt::Hold;
use crate::rt::held::Held;
use crate::rt::pool::Pool;

/// The resume error of a connection-type mismatch: the C API's `RVG_RESUME_ERROR_INTERNAL_FAULT`.
pub const RESUME_ERROR_INTERNAL_FAULT: u64 = 0;
/// A binary version the device does not support: `RVG_RESUME_ERROR_INVALID_GUEST_VERSION`.
pub const RESUME_ERROR_INVALID_GUEST_VERSION: u64 = 3;
/// A host lacking what the guest asked for: `RVG_RESUME_ERROR_INCOMPATIBLE_DEVICE`.
pub const RESUME_ERROR_INCOMPATIBLE_DEVICE: u64 = 4;
/// Another display port count: `RVG_RESUME_ERROR_INVALID_DISPLAY_PORT_COUNT`.
pub const RESUME_ERROR_INVALID_DISPLAY_PORT_COUNT: u64 = 5;

/// The first binary version a suspend blob may not carry.
const RESUME_BINARY_VERSION_LIMIT: u32 = 0xf6;

/// The device-info max key after a reset.
const RESET_DEVICE_INFO_MAX_KEY: i32 = 1;

/// The child FIFO slots; the walks run over 1..64 (slot 0 is the root FIFO's id).
const CHILD_FIFO_SLOTS: u32 = 64;

/// The task words of the root node.
const ROOT_TASK_WORDS: u32 = 256;

/// Where the first child FIFO node's in-use halfword sits in the root node.
const CHILD_NODE_FLAG: u64 = 0x40e;
/// The child FIFO node stride.
const CHILD_NODE_STRIDE: u64 = 0x14;

/// The bytes of a task root node a resume reads.
const TASK_ROOT_NODE_SIZE: usize = 0x1c;
/// The task length's offset in the task root node.
const TASK_ROOT_LENGTH: usize = 0x14;

/// Stores the resume error `code` when the caller gave somewhere to store it.
fn set_error(error: Option<&mut Option<u64>>, code: u64) {
    if let Some(slot) = error {
        *slot = Some(code);
    }
}

impl<P: Platform> DeviceCore<P> {
    /// The root FIFO, read now without its lock.
    fn current_root_fifo(&self) -> Option<Arc<Fifo<P>>> {
        self.root_fifo.atomics().root_fifo()
    }

    /// The FIFO in child slot `index`, read now.
    fn current_child_fifo(&self, index: u32) -> Option<Arc<Fifo<P>>> {
        self.child_fifos
            .lock()
            .get(index.to_usize())
            .cloned()
            .flatten()
    }

    /// The guest display at `port`, read without the displays' lock: its slot is read once for the
    /// test and once for the use, so neither waits behind a page 0 or a detach holding the lock.
    /// `None` when either read saw no guest display.
    fn current_guest_display(&self, port: u64) -> Option<Arc<GuestDisplay<P>>> {
        let views = self.displays.atomics();
        if !views.has_guest_display(port) {
            return None;
        }
        views.guest_display(port)
    }

    /// The display port count.
    fn port_count(&self) -> u64 {
        u64::from(self.display_port_count.load(Ordering::Relaxed))
    }

    /// The EFI registers, read without the displays' lock.
    fn efi_registers(&self) -> EfiRegisters {
        let e = self.displays.atomics();
        EfiRegisters {
            enabled: e.enabled.load(Ordering::Relaxed),
            start: e.start.load(Ordering::Relaxed),
            length: e.length.load(Ordering::Relaxed),
            depth: e.depth.load(Ordering::Relaxed),
            mode: e.mode.load(Ordering::Relaxed),
            stride: e.stride.load(Ordering::Relaxed),
        }
    }

    /// Enters `EnteringSuspend` and quiesces every FIFO, the root first. Dies with a FIFO in
    /// child slot 0.
    ///
    /// Stamp waiters are not woken: a FIFO blocked in a stamp wait stays blocked
    /// (`tests::suspend_wakeups::will_suspend_leaves_waiters_blocked`).
    pub fn will_suspend(&self) {
        log::info!("device suspending");
        self.run_state.store(RunState::EnteringSuspend);
        if let Some(root) = self.current_root_fifo() {
            root.quiesce();
        }
        crate::fatal_unless!(
            self.current_child_fifo(0).is_none(),
            DeviceDeath::ChildFifoInSlotZero
        );
        for index in 1..CHILD_FIFO_SLOTS {
            if let Some(child) = self.current_child_fifo(index) {
                child.quiesce();
            }
        }
    }

    /// Finishes a suspend `will_suspend` began (dies otherwise): drains and drops every child,
    /// drains the root FIFO, flushes and tears down every task, serializes the device, drops the
    /// root FIFO and pauses the guest displays. The root node, the root task, the guest displays,
    /// EFI and the fault mask survive. The blob is always given: whatever the guest stored, it
    /// serializes.
    ///
    /// The root FIFO's `enabled` register stays as it was with no root FIFO left
    /// (`tests::finish_suspend::serializes`).
    pub fn finish_suspend(&self) -> Option<Vec<u8>> {
        crate::fatal_unless!(
            self.run_state.load() == RunState::EnteringSuspend,
            DeviceDeath::SuspendNotBegun
        );
        self.run_state.store(RunState::Suspended);
        self.signal_fifos();
        for index in 1..CHILD_FIFO_SLOTS {
            if let Some(fifo) = self.current_child_fifo(index) {
                fifo.drain();
                // The slot's reference goes after the drained one: the slot is emptied first.
                let slot = self
                    .child_fifos
                    .lock()
                    .get_mut(index.to_usize())
                    .and_then(Option::take);
                drop(fifo);
                drop(slot);
            }
        }
        if let Some(root) = self.current_root_fifo() {
            root.drain();
        }
        {
            // Ours: the detached heaps made so far release their GPU VA ranges as they drop,
            // which a resume in this process has the guest pin again.
            crate::resources::heap::detach::tasks_torn_down(&self.platform);
            let mut tasks = self.tasks.lock();
            // A real pool and a holder around the walk, under the task lock. What the tasks'
            // flushes hold deep in the walk (a page-table read's device handle), under the walk's
            // hold, goes when the holder drops, and what the flushes and teardowns autorelease
            // (the paging command buffer and its encoders) when the pool pops just after: both
            // before the table empties. The walk's copy of the tasks goes after them.
            let pool = Pool::push(self.platform.as_ref());
            let held = Held::new();
            let walked: Vec<_> = tasks.values().cloned().collect();
            let walk = Hold::of(&held);
            for task in &walked {
                // A raise here has no handler: it ends the process.
                if let Err(r) = task.flush_resources(walk) {
                    r.fatal()
                }
                task.teardown();
            }
            drop(held);
            drop(pool);
            drop(walked);
            drop(tasks.drain());
        }
        let state = self.serialize_suspend_state();
        *self.stamp_memory.lock() = None;
        let (root, mapping) = {
            let mut r = self.root_fifo.lock();
            (r.root_fifo.take(), r.root_fifo_mapping.take())
        };
        drop(root);
        drop(mapping);
        for port in 0..self.port_count() {
            if let Some(guest_display) = self.current_guest_display(port) {
                guest_display.pause();
            }
        }
        log::info!("device suspended: {} bytes of state", state.len());
        Some(state)
    }

    /// The device's registers as a suspend blob, with the device info the guest last asked for
    /// (its max key, whatever the guest stored). The root FIFO's written, read and start offsets
    /// and its fault offset are 0 without a root FIFO.
    pub fn serialize_suspend_state(&self) -> Vec<u8> {
        // Every register read without a lock, but the root FIFO's base page and length, which
        // have no unlocked view (module doc).
        let root_page_number = self
            .root_task
            .atomics()
            .root_page_number
            .load(Ordering::Relaxed);
        let fifo = self.root_fifo.lock().fifo;
        let fifo_enabled = self.root_fifo.atomics().enabled();
        let root = self.current_root_fifo();
        let efi = self.efi_registers();
        // Without a root FIFO its registers read 0.
        let state = SuspendDeviceState {
            root_page_number,
            binary_version: self.binary_version.load(Ordering::Relaxed).cast_unsigned(),
            fifo_base_page: fifo.base_page,
            fifo_enabled,
            fifo_length: fifo.length,
            fifo_written: root.as_ref().and_then(|r| r.root_written()).unwrap_or(0),
            fifo_read: root.as_ref().and_then(|r| r.root_read()).unwrap_or(0),
            fifo_start: root
                .as_ref()
                .and_then(|r| r.root_start_offset())
                .unwrap_or(0),
            interrupt_display: self.interrupt_display.load(Ordering::SeqCst),
            interrupt_stamp: self.interrupt_stamp.load(Ordering::SeqCst),
            interrupt_fault: u32::from(self.interrupt_fault_mask.load(Ordering::SeqCst) != 0),
            efi_enabled: efi.enabled,
            efi_start: efi.start,
            efi_length: efi.length,
            efi_depth: efi.depth,
            efi_mode: efi.mode,
            efi_stride: efi.stride,
            display_port_count: self.display_port_count.load(Ordering::Relaxed),
            fifo_fault_offset: root.as_ref().map_or(0, |r| r.fault_offset()),
            display_port_connection_types: self
                .display_port_connection_types
                .load(Ordering::Relaxed),
        };
        let max_key = self.guest_device_info_max_key.load(Ordering::Relaxed);
        let mut pairs = [KeyValuePair::default(); SERIALIZED_INFO_PAIRS];
        let written = write_device_info(
            OnFault::Die,
            &self.mtl_device.caps(),
            &self.info_snapshot(),
            max_key.cast_unsigned(),
            SERIALIZED_INFO_PAIRS.low_u32(),
            |i, pair| {
                if let Some(slot) = pairs.get_mut(i.to_usize()) {
                    *slot = pair;
                }
            },
        );
        // A Metal raise here has no handler: it ends the process.
        if let Err(t) = written {
            t.fatal()
        }
        SuspendState::new(state, max_key, &pairs).serialize()
    }

    /// Whether the device offers what the blob's device info asks for; stores
    /// `IncompatibleDevice` when it does not.
    ///
    /// The host's list is the one at the blob's version, just set, with this device's
    /// setup-dependent bits (`tests::will_resume::capabilities_at_blob_version`).
    fn check_suspend_device_info(
        &self,
        state: &SuspendState,
        error: Option<&mut Option<u64>>,
    ) -> bool {
        match check_capabilities(
            &self.mtl_device.caps(),
            &self.info_snapshot(),
            &state.device_info,
        ) {
            Ok(true) => return true,
            Ok(false) => {}
            Err(t) => t.fatal(),
        }
        // The check's own line names the capability.
        log::warn!("resume refused: this host lacks what the suspended guest asked for");
        set_error(error, RESUME_ERROR_INCOMPATIBLE_DEVICE);
        false
    }

    /// Validates a suspend state and stores its registers, for [`DeviceCore::did_resume`] to
    /// rebuild from. `false` for a refusal.
    ///
    /// `state` is `None` for no state (the C API's NULL): the device refuses it with
    /// `InvalidContent`, a path apart from validating empty data.
    ///
    /// `error` is where the caller wants the resume error code (`None` for a NULL error pointer):
    /// without it no code is stored, and the binary version refusals log nothing. A `false`
    /// answer does not imply a stored code.
    ///
    /// `LeavingSuspendEarly` is set first and kept on every refusal
    /// (`tests::will_resume::nil_state`). The blob's binary version is applied before the later
    /// checks (`tests::will_resume::port_count_mismatch`).
    pub fn will_resume(
        self: &Arc<Self>,
        state: Option<&[u8]>,
        mut error: Option<&mut Option<u64>>,
    ) -> bool {
        log::info!("device resuming");
        self.run_state.store(RunState::LeavingSuspendEarly);
        // Ours: the platform keeps the guest's freed GPU VA ranges for its pins.
        self.platform.resuming();
        let Some(bytes) = state else {
            set_error(error, RESUME_ERROR_INVALID_CONTENT.cast_unsigned());
            return false;
        };
        // A refusal stores its code only where the caller gave somewhere to store it.
        let blob = match SuspendState::parse(bytes) {
            Ok(blob) => blob,
            Err(e) => {
                set_error(error, e.code.cast_unsigned());
                return false;
            }
        };
        let device_state = blob.registers;
        // The line is logged only when the caller gave somewhere to store the code
        // (`tests::will_resume::version_too_high`).
        if device_state.binary_version >= RESUME_BINARY_VERSION_LIMIT {
            if let Some(slot) = error {
                log::warn!(
                    "resume refused: the state's binary version {:#x} is not below {:#x}",
                    device_state.binary_version,
                    RESUME_BINARY_VERSION_LIMIT
                );
                *slot = Some(RESUME_ERROR_INVALID_GUEST_VERSION);
            }
            return false;
        }
        self.set_binary_version(device_state.binary_version);
        if self.binary_version.load(Ordering::Relaxed) < 0 {
            if let Some(slot) = error {
                log::warn!(
                    "resume refused: the state's binary version {:#x} is not supported",
                    device_state.binary_version
                );
                *slot = Some(RESUME_ERROR_INVALID_GUEST_VERSION);
            }
            return false;
        }
        if !self.check_suspend_device_info(&blob, error.as_deref_mut()) {
            return false;
        }
        let port_count = self.display_port_count.load(Ordering::Relaxed);
        if port_count != device_state.display_port_count {
            log::warn!(
                "resume refused: the state has {} display ports, and this device {port_count}",
                device_state.display_port_count
            );
            set_error(error, RESUME_ERROR_INVALID_DISPLAY_PORT_COUNT);
            return false;
        }
        // A connection-type mismatch is InternalFault (0)
        // (`tests::will_resume::connection_type_mismatch`).
        if self.features.load().supports_display_port_connection_type {
            let types = self.display_port_connection_types.load(Ordering::Relaxed);
            if types != device_state.display_port_connection_types {
                log::warn!(
                    "resume refused: the state's display port connection types are {:#x}, and \
                     this device's {types:#x}",
                    device_state.display_port_connection_types
                );
                set_error(error, RESUME_ERROR_INTERNAL_FAULT);
                return false;
            }
        }
        // The root page and the root FIFO's `enabled` through their unlocked views.
        self.root_task
            .atomics()
            .root_page_number
            .store(device_state.root_page_number, Ordering::Relaxed);
        self.root_fifo
            .atomics()
            .enabled
            .store(device_state.fifo_enabled, Ordering::Relaxed);
        // The root FIFO's other registers have no unlocked view (module doc).
        {
            let mut r = self.root_fifo.lock();
            r.fifo.base_page = device_state.fifo_base_page;
            r.fifo.length = device_state.fifo_length;
            r.fifo.start = device_state.fifo_start;
            r.fifo.read = device_state.fifo_read;
            r.fifo.written = device_state.fifo_written;
            r.fifo.fault_offset = device_state.fifo_fault_offset;
        }
        self.interrupt_display
            .store(device_state.interrupt_display, Ordering::SeqCst);
        self.interrupt_stamp
            .store(device_state.interrupt_stamp, Ordering::SeqCst);
        self.interrupt_fault_mask.store(0, Ordering::SeqCst);
        {
            let e = self.displays.atomics();
            e.enabled.store(device_state.efi_enabled, Ordering::Relaxed);
            e.start.store(device_state.efi_start, Ordering::Relaxed);
            e.length.store(device_state.efi_length, Ordering::Relaxed);
            e.depth.store(device_state.efi_depth, Ordering::Relaxed);
            e.mode.store(device_state.efi_mode, Ordering::Relaxed);
            e.stride.store(device_state.efi_stride, Ordering::Relaxed);
        }
        self.guest_device_info_max_key
            .store(blob.device_info_max_key, Ordering::Relaxed);
        true
    }

    /// Rebuilds the device from the registers `will_resume` stored: EFI, the root page (tearing
    /// down a live root first), the root FIFO, the children the root node marks, the tasks and
    /// the display pages the root node names now; then starts the FIFOs and interrupts once.
    ///
    /// A live root is torn down by the root page's store, which clears the root FIFO's
    /// `enabled`: resuming in place loses the root FIFO
    /// (`tests::did_resume_fifos::in_place_loses_root_fifo`). Tasks are rebuilt from the root
    /// words as they are now (`tests::did_resume_rebuild::rebuilds_tasks_and_displays`).
    pub fn did_resume(self: &Arc<Self>, hold: Hold<'_>) {
        log::info!("device resume continues");
        self.run_state.store(RunState::Resuming);
        // Each register read without a lock, as its call is made.
        let e = self.displays.atomics();
        let efi_enabled = e.enabled.load(Ordering::Relaxed);
        if efi_enabled != 0 {
            self.set_efi_display(efi_enabled);
            let start = e.start.load(Ordering::Relaxed);
            self.set_efi_framebuffer_start(start);
            let length = e.length.load(Ordering::Relaxed);
            self.set_efi_framebuffer_length(length);
            let depth = e.depth.load(Ordering::Relaxed);
            self.set_efi_framebuffer_depth(depth);
            let mode = e.mode.load(Ordering::Relaxed);
            self.set_efi_framebuffer_mode(mode);
            let stride = e.stride.load(Ordering::Relaxed);
            self.set_efi_framebuffer_stride(stride);
        }
        let root_page_number = self
            .root_task
            .atomics()
            .root_page_number
            .load(Ordering::Relaxed);
        self.set_root_page(hold, root_page_number);
        let root_node = self.root_node.atomics().root_node();
        if let Some(root_node) = root_node {
            self.resume_root_fifo();
            self.resume_children(root_node);
            self.resume_tasks(hold, root_node);
            self.resume_displays(hold, root_node);
        }
        self.run_state.store(RunState::Running);
        for port in 0..self.port_count() {
            if let Some(guest_display) = self.current_guest_display(port) {
                guest_display.finish_restore(hold);
            }
        }
        self.start_fifos();
        self.interrupt();
        log::info!("device resumed");
    }

    /// The resume's root FIFO: created again when its registers enable it (dies when it cannot
    /// be), with its read and written offsets and fault offset.
    ///
    /// A restored fault offset sets the register and mask bit 0 only
    /// (`tests::did_resume_fifos::restored_fault_offset`).
    fn resume_root_fifo(&self) {
        if self.root_fifo.atomics().enabled() == 0 {
            return;
        }
        self.create_root_fifo();
        let root = self.current_root_fifo();
        crate::fatal_unless!(root.is_some(), DeviceDeath::ResumedRootFifoMissing);
        let Some(root) = root else {
            return;
        };
        // The read and written offsets and the fault offset have no unlocked view.
        let fifo = self.root_fifo.lock().fifo;
        root.set_root_read(fifo.read);
        root.set_root_written(fifo.written);
        if fifo.fault_offset != u32::MAX {
            root.set_fault_offset(fifo.fault_offset);
            self.interrupt_fault_mask.fetch_or(1, Ordering::SeqCst);
        }
    }

    /// The resume's children: each child node the root node marks in use is created again.
    ///
    /// Children are created on the calling thread, off the FIFO queue, and only below the current
    /// `maxFIFOCount` (`tests::did_resume_fifos::children_off_queue_below_max`). The walk's
    /// length is `maxFIFOCount − 1` as a `u32`: with `maxFIFOCount` 0 it leaves the root page,
    /// and dies there (`tests::did_resume_rebuild::zero_fifo_count_walks_off_page`). The restored
    /// fault bits are by child id
    /// (`tests::did_resume_fifos::child_refused_fault_bits_and_start_failure`). Nothing here
    /// requires the root FIFO: with the blob's root FIFO disabled the children come back and
    /// start while the stamp memory is NULL, and a child's first stamp signal is the host death
    /// (`tests::did_resume_fifos::child_without_root_fifo_signals_through_null`).
    fn resume_children(&self, root_node: GuestPtr) {
        let max_fifo_count = self.features.load().max_fifo_count;
        if max_fifo_count == 1 {
            return;
        }
        let walk = u64::from(max_fifo_count.wrapping_sub(1));
        for index in 0..walk {
            let offset = CHILD_NODE_FLAG.wrapping_add(index.wrapping_mul(CHILD_NODE_STRIDE));
            crate::fatal_unless!(
                offset.wrapping_add(2) <= APV_PAGE_SIZE,
                DeviceDeath::ResumeWalkLeftRootPage
            );
            if root_node.read_u16(offset, DeviceDeath::RootNodeFault) != 0 {
                let fifo_id = index.low_u32().wrapping_add(1);
                // The create's own line says why.
                if !self.create_fifo_internal(fifo_id, true) {
                    log::warn!("resume: child FIFO {fifo_id} not created again");
                }
            }
        }
        for index in 1..max_fifo_count {
            if let Some(fifo) = self.current_child_fifo(index)
                && fifo.fault_offset() != u32::MAX
            {
                self.interrupt_fault_mask
                    .fetch_or(crate::num::bit64(index), Ordering::SeqCst);
            }
        }
    }

    /// The resume's tasks, each under a handler of its own: a raise is logged and the walk goes
    /// on.
    ///
    /// A failed read of a task root node leaves a zeroed buffer, so such a task is defined with
    /// length 0 (`tests::did_resume_tasks::unreadable_task_root`).
    fn resume_tasks(&self, hold: Hold<'_>, root_node: GuestPtr) {
        for task_id in 0..ROOT_TASK_WORDS {
            let word_offset = u64::from(task_id).wrapping_mul(4);
            let resumed = (|| -> RaiseResult<()> {
                let task_root = root_node.read_u32(word_offset, DeviceDeath::RootNodeFault);
                if task_root != 0 {
                    let mut node = [0u8; TASK_ROOT_NODE_SIZE];
                    let read = self.read_from_physical_address(
                        OnFault::Return,
                        u64::from(task_root).wrapping_shl(APV_PAGE_SHIFT),
                        &mut node,
                    )?;
                    if !read {
                        log::warn!(
                            "resume: task {task_id}'s root node at page {task_root:#x} not \
                             readable, so its length reads 0"
                        );
                    }
                    let mut length = [0u8; 8];
                    if let Some(bytes) = node.get(TASK_ROOT_LENGTH..TASK_ROOT_NODE_SIZE) {
                        length.copy_from_slice(bytes);
                    }
                    // The root word is read again for the define.
                    self.create_task_id(
                        hold,
                        task_id,
                        root_node.read_u32(word_offset, DeviceDeath::RootNodeFault),
                        u64::from_le_bytes(length),
                        true,
                    )?;
                }
                Ok(())
            })();
            if let Err(e) = resumed {
                log::warn!("resume: task {task_id} not set up: {}", e.exception());
                hold.land(e);
            }
        }
    }

    /// The resume's display pages, each port under a handler of its own: a raise is logged and
    /// the walk goes on.
    fn resume_displays(&self, hold: Hold<'_>, root_node: GuestPtr) {
        for port in 0..self.port_count() {
            let resumed = (|| -> RaiseResult<()> {
                let page = root_node.read_u32(
                    crate::device::displays::ROOT_DISPLAY_WORDS
                        .wrapping_add(port)
                        .wrapping_mul(4),
                    DeviceDeath::RootNodeFault,
                );
                if page != 0 {
                    self.set_shared_display_state_page(page, port, true)?;
                }
                Ok(())
            })();
            if let Err(e) = resumed {
                log::warn!("resume: display port {port} not set up: {}", e.exception());
                hold.land(e);
            }
        }
    }

    /// Starts every child, then the root FIFO, logging each refusal (a resume's and an
    /// unpause's). A root FIFO's start never refuses: its mapping is set up and its thread
    /// exists once it has started.
    fn start_fifos(&self) {
        for index in 1..CHILD_FIFO_SLOTS {
            if let Some(fifo) = self.current_child_fifo(index)
                && !fifo.start(&self.platform)
            {
                log::warn!("child FIFO {index} did not start");
            }
        }
        if let Some(root) = self.current_root_fifo()
            && !root.start(&self.platform)
        {
            log::warn!("root FIFO did not start");
        }
    }

    /// Quiesces and drains the root FIFO, then the children, pauses the guest displays, and only
    /// then enters `Paused`: the drains publish stamps and raise interrupts during the call.
    ///
    /// The paused state is set last (`tests::pause::pauses`).
    pub fn pause(&self) {
        if let Some(root) = self.current_root_fifo() {
            root.quiesce();
        }
        self.signal_fifos();
        if let Some(root) = self.current_root_fifo() {
            root.drain();
        }
        for index in 1..CHILD_FIFO_SLOTS {
            if let Some(child) = self.current_child_fifo(index) {
                child.quiesce();
            }
        }
        self.signal_fifos();
        for index in 1..CHILD_FIFO_SLOTS {
            if let Some(child) = self.current_child_fifo(index) {
                child.drain();
            }
        }
        for port in 0..self.port_count() {
            if let Some(guest_display) = self.current_guest_display(port) {
                guest_display.pause();
            }
        }
        self.run_state.store(RunState::Paused);
    }

    /// Running again: unpauses the guest displays, starts every FIFO on the FIFO queue
    /// (synchronously), and interrupts once.
    pub fn unpause(&self) {
        self.run_state.store(RunState::Running);
        for port in 0..self.port_count() {
            if let Some(guest_display) = self.current_guest_display(port) {
                // A raise here has no handler, in the device or the host: it ends the process.
                if let Err(raised) = guest_display.unpause() {
                    raised.fatal();
                }
            }
        }
        // The queued job holds a retain of the device.
        let device = self.weak_self.upgrade();
        self.fifo_queue.sync(move || {
            if let Some(device) = device {
                device.start_fifos();
            }
        });
        self.interrupt();
    }

    /// Stops the device: the root is torn down as a root page of 0 does.
    pub fn stop(&self, hold: Hold<'_>) {
        self.set_root_page(hold, 0);
    }

    /// Tears the root down, returns to binary version 0 (logging it as unknown, and dying at
    /// feature level 5 or more), turns EFI off and sets the device-info max key to 1.
    pub fn reset(self: &Arc<Self>, hold: Hold<'_>) {
        log::info!("device reset");
        self.set_root_page(hold, 0);
        self.set_binary_version(0);
        self.set_efi_display(0);
        self.guest_device_info_max_key
            .store(RESET_DEVICE_INFO_MAX_KEY, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_support::{Lines, Opts, TestDevice, device};
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;
    use crate::testkit::capture::catch_death;

    fn dev() -> TestDevice {
        device(Opts::default())
    }

    fn died(r: Result<impl std::fmt::Debug, DeathSite>, death: DeviceDeath) {
        assert_eq!(r.expect_err("expected a death"), death.site());
    }

    /// A blob of `state` with an empty device-info list and max key 1, as a guest that never
    /// asked leaves it.
    fn blob(state: &SuspendDeviceState) -> Vec<u8> {
        SuspendState::new(*state, 1, &[]).serialize()
    }

    /// A state this test device accepts (2 ports, the given version).
    fn state(binary_version: u32) -> SuspendDeviceState {
        SuspendDeviceState {
            root_page_number: 0x21,
            binary_version,
            fifo_base_page: 0x30,
            fifo_enabled: 1,
            fifo_length: 0x4000,
            fifo_written: 0x40,
            fifo_read: 0x20,
            fifo_start: 0x10,
            interrupt_display: 3,
            interrupt_stamp: 5,
            interrupt_fault: 1,
            efi_enabled: 0,
            efi_start: 7,
            efi_length: 8,
            efi_depth: 32,
            efi_mode: 1,
            efi_stride: 0x100,
            display_port_count: 2,
            fifo_fault_offset: 0x44,
            display_port_connection_types: 0,
        }
    }

    mod will_suspend {
        use super::*;

        /// Enters EnteringSuspend (no FIFOs to quiesce)
        #[test]
        fn enters_suspend() {
            let t = dev();
            t.device.will_suspend();
            assert_eq!(t.device.run_state.load(), RunState::EnteringSuspend);
        }

        /// A FIFO in child slot 0 dies
        #[test]
        fn child_slot_zero_dies() {
            let t = dev().with_root_page();
            // child node 1 (root + 0x400) names stamp index 1 (u16 at +0xc)
            t.set_guest_root_word((0x400 + 0xc) / 4, 1);
            assert!(t.device.create_fifo_internal(1, true));
            {
                let mut children = t.device.child_fifos.lock();
                let fifo = children[1].take();
                children[0] = fifo;
            }
            died(
                catch_death(|| t.device.will_suspend()),
                DeviceDeath::ChildFifoInSlotZero,
            );
        }
    }

    mod finish_suspend {
        use super::*;

        /// Without willSuspend it dies
        #[test]
        fn requires_entering_suspend() {
            let t = dev();
            died(
                catch_death(|| t.device.finish_suspend()),
                DeviceDeath::SuspendNotBegun,
            );
        }

        /// Serializes with the root FIFO's `enabled` kept and no root FIFO left
        #[test]
        fn serializes() {
            let t = dev();
            t.device.root_fifo.lock().fifo.enabled = 1;
            t.device.will_suspend();
            let state = t.device.finish_suspend().unwrap();
            assert_eq!(t.device.run_state.load(), RunState::Suspended);
            let s = SuspendState::parse(&state).unwrap();
            assert_eq!(s.registers.fifo_enabled, 1);
            assert_eq!(t.device.root_fifo.lock().fifo.enabled, 1);
            assert!(t.device.root_fifo.lock().root_fifo.is_none());
        }

        /// The walk's pool is a real one, pushed and popped under the task lock, before the
        /// table empties: the tasks' last releases (their `destroyTask`) come after it pops
        #[test]
        fn the_walk_s_pool_pops_before_the_table_empties() {
            use crate::device::root::rig::rig_with;
            use crate::rt::pool::test_native::{events, note, with_test_native};
            let r = rig_with(|blocks| {
                let destroy = blocks.destroy_task.clone().expect("destroyTask");
                blocks.destroy_task = Some(Arc::new(move |t| {
                    note("destroyTask");
                    destroy(t)
                }));
            });
            r.device.set_root_page(Hold::NONE, 1);
            let pa = 0x40u64 * APV_PAGE_SIZE;
            r.platform.fake_vm().sync();
            assert!(r.host.ram().write_u32(pa + 4, 1));
            assert!(r.host.ram().write_u64(pa + 0x14, 0x4000));
            r.platform.fake_vm().sync();
            {
                for id in 1..3 {
                    r.device
                        .create_task_id(Hold::NONE, id, 0x40, 0x4000, true)
                        .unwrap();
                }
            }
            r.device.will_suspend();
            let mut got = Vec::new();
            with_test_native(|| {
                let _ = events();
                assert!(r.device.finish_suspend().is_some());
                got = events();
            });
            assert!(r.device.tasks.lock().is_empty());
            assert_eq!(
                got,
                ["push 0x100", "pop 0x100", "destroyTask", "destroyTask"]
            );
        }
    }

    mod serialize_suspend_state {
        use super::*;

        /// The device's registers, the root FIFO's offsets zero without one, the fault flag the
        /// mask's
        #[test]
        fn reads_device() {
            let t = dev();
            t.device.root_task.lock().root_page_number = 0x21;
            t.device.binary_version.store(0xf5, Ordering::Relaxed);
            t.device.root_fifo.lock().fifo.base_page = 0x30;
            t.device.interrupt_fault_mask.store(0x10, Ordering::SeqCst);
            t.device.displays.lock().efi.stride = 0x100;
            let blob = t.device.serialize_suspend_state();
            let ds = SuspendState::parse(&blob).unwrap().registers;
            assert_eq!(ds.root_page_number, 0x21);
            assert_eq!(ds.binary_version, 0xf5);
            assert_eq!(ds.fifo_base_page, 0x30);
            assert_eq!(ds.interrupt_fault, 1);
            assert_eq!(ds.efi_stride, 0x100);
            assert_eq!(ds.display_port_count, 2);
            assert_eq!((ds.fifo_written, ds.fifo_read, ds.fifo_start), (0, 0, 0));
            assert_eq!(ds.fifo_fault_offset, 0);
        }
    }

    mod will_resume {
        use super::*;

        /// A nil state: InvalidContent, run state 3 kept
        #[test]
        fn nil_state() {
            let t = dev();
            let mut e = None;
            assert!(!t.device.will_resume(None, Some(&mut e)));
            assert_eq!(e, Some(2));
            assert_eq!(t.device.run_state.load(), RunState::LeavingSuspendEarly);
            assert!(!t.device.will_resume(None, None));
        }

        /// A blob that fails validation gives validate's code
        #[test]
        fn invalid_blob() {
            let t = dev();
            let mut e = None;
            assert!(!t.device.will_resume(Some(&[1, 2, 3]), Some(&mut e)));
            assert_eq!(e, Some(2));
            let mut bad_version = blob(&state(0xf5));
            bad_version[4] = 2;
            let mut e = None;
            assert!(!t.device.will_resume(Some(&bad_version), Some(&mut e)));
            assert_eq!(e, Some(1));
        }

        /// A version from 0xf6 up is refused, logging only with an error
        #[test]
        fn version_too_high() {
            let t = dev();
            let lines = Lines::capture();
            let b = blob(&state(0xf6));
            let refusal = "resume refused: the state's binary version 0xf6 is not below 0xf6";
            assert!(!t.device.will_resume(Some(&b), None));
            assert_eq!(lines.count(refusal), 0);
            let mut e = None;
            assert!(!t.device.will_resume(Some(&b), Some(&mut e)));
            assert_eq!(e, Some(RESUME_ERROR_INVALID_GUEST_VERSION));
            assert_eq!(lines.count(refusal), 1);
            assert_eq!(t.device.binary_version.load(Ordering::Relaxed), 0);
        }

        /// A port-count mismatch is refused after the version is applied
        #[test]
        fn port_count_mismatch() {
            let t = dev();
            let mut s = state(0xf5);
            s.display_port_count = 1;
            let mut e = None;
            assert!(!t.device.will_resume(Some(&blob(&s)), Some(&mut e)));
            assert_eq!(e, Some(RESUME_ERROR_INVALID_DISPLAY_PORT_COUNT));
            assert_eq!(t.device.binary_version.load(Ordering::Relaxed), 0xf5);
        }

        /// A connection-type mismatch is InternalFault (0)
        #[test]
        fn connection_type_mismatch() {
            let t = dev();
            let mut s = state(0xf5);
            s.display_port_connection_types = 1;
            let mut e = None;
            assert!(!t.device.will_resume(Some(&blob(&s)), Some(&mut e)));
            assert!(
                t.device
                    .features
                    .load()
                    .supports_display_port_connection_type
            );
            assert_eq!(e, Some(RESUME_ERROR_INTERNAL_FAULT));
        }

        /// The registers are stored, the fault mask cleared, the max key from the blob
        #[test]
        fn stores_registers() {
            let t = dev();
            t.device.interrupt_fault_mask.store(9, Ordering::SeqCst);
            let mut e = None;
            assert!(
                t.device
                    .will_resume(Some(&blob(&state(0xf5))), Some(&mut e))
            );
            assert_eq!(e, None);
            let d = &t.device;
            assert_eq!(d.root_task.lock().root_page_number, 0x21);
            let f = d.root_fifo.lock().fifo;
            assert_eq!(
                (
                    f.enabled,
                    f.base_page,
                    f.length,
                    f.start,
                    f.read,
                    f.written,
                    f.fault_offset
                ),
                (1, 0x30, 0x4000, 0x10, 0x20, 0x40, 0x44)
            );
            assert_eq!(d.interrupt_display.load(Ordering::SeqCst), 3);
            assert_eq!(d.interrupt_stamp.load(Ordering::SeqCst), 5);
            assert_eq!(d.interrupt_fault_mask.load(Ordering::SeqCst), 0);
            assert_eq!(d.displays.lock().efi.stride, 0x100);
            assert_eq!(d.guest_device_info_max_key.load(Ordering::Relaxed), 1);
            assert_eq!(d.run_state.load(), RunState::LeavingSuspendEarly);
        }

        /// The guest's list is checked against the host's at the blob's binary version, just
        /// set: key 12 (dualPlaneTextures) = 1 is refused at version 40 (IncompatibleDevice) even
        /// on a device running 0xf5, and passes at 0xf5
        #[test]
        fn capabilities_at_blob_version() {
            let with_key_12 = |version: u32| {
                let pairs = [KeyValuePair { key: 12, value: 1 }];
                SuspendState::new(state(version), 13, &pairs).serialize()
            };
            let t = dev();
            t.device.set_binary_version(0xf5);
            assert!(t.device.features.load().dual_plane_textures);
            let mut e = None;
            assert!(!t.device.will_resume(Some(&with_key_12(40)), Some(&mut e)));
            assert_eq!(e, Some(RESUME_ERROR_INCOMPATIBLE_DEVICE));
            assert_eq!(t.device.binary_version.load(Ordering::Relaxed), 40);
            let t = dev();
            let mut e = None;
            assert!(t.device.will_resume(Some(&with_key_12(0xf5)), Some(&mut e)));
            assert_eq!(e, None);
        }

        /// The device-info max key comes back as the blob carries it, whatever the guest
        /// stored: a negative one included
        #[test]
        fn device_info_max_key_round_trips() {
            for max_key in [i32::MIN, -1, 0, 47, i32::MAX] {
                let t = dev();
                let b = SuspendState::new(state(0xf5), max_key, &[]).serialize();
                assert!(t.device.will_resume(Some(&b), None));
                assert_eq!(
                    t.device.guest_device_info_max_key.load(Ordering::Relaxed),
                    max_key
                );
            }
        }

        /// A blob whose magic is not ours (`PGSD`, then key-value pairs) is refused as
        /// InvalidContent, with nothing stored
        #[test]
        fn pgsd_blob_is_refused() {
            let t = dev();
            let mut words: Vec<u32> = vec![0x5047_5344, 1];
            for (k, v) in [(2u32, 0xf5u32), (18, 2)] {
                words.extend([k, v]);
            }
            words.extend([0, 0]);
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let mut e = None;
            assert!(!t.device.will_resume(Some(&bytes), Some(&mut e)));
            assert_eq!(e, Some(2));
            assert_eq!(t.device.binary_version.load(Ordering::Relaxed), 0);
            assert_eq!(t.device.run_state.load(), RunState::LeavingSuspendEarly);
        }
    }

    mod did_resume {
        use super::*;

        /// No EFI and no root page: running again, one interrupt
        #[test]
        fn nothing_to_rebuild() {
            let t = dev();
            t.device.did_resume(Hold::NONE);
            assert_eq!(t.device.run_state.load(), RunState::Running);
            assert_eq!(t.host.interrupts(), vec![0]);
        }
    }

    mod did_resume_rebuild {
        use super::*;
        use crate::device::displays::ROOT_DISPLAY_WORDS;
        use crate::device::test_support::ROOT_PAGE;

        /// Tasks and display pages come from the root words as they are
        #[test]
        fn rebuilds_tasks_and_displays() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            // task 2's root node at page 0x20 names a 16 KiB task; task 3's is empty (length 0)
            t.task_root_node(0x20, 0x4000);
            t.set_guest_root_word(2, 0x20);
            t.set_guest_root_word(3, 0x21);
            t.set_guest_root_word(ROOT_DISPLAY_WORDS + 1, 0x10);
            let lines = Lines::capture();
            t.device.did_resume(Hold::NONE);
            assert_eq!(t.device.run_state.load(), RunState::Running);
            assert_eq!(t.device.tasks.lock().len(), 1);
            assert!(t.device.tasks.lock().get(2).is_some());
            // The raise's detail.
            assert_eq!(
                lines.count("resume: task 3 not set up: task 3 not created"),
                1
            );
            let guest_display = t.device.get_guest_display_at_port(1).unwrap();
            assert!(!guest_display.guest_display_mutex.lock().will_restore);
            assert_eq!(t.host.interrupts(), vec![0]);
        }

        /// The root FIFO the blob enables must come back: refused parameters die
        #[test]
        fn root_fifo_refused_dies() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            t.device.root_fifo.lock().fifo.enabled = 1;
            died(
                catch_death(|| t.device.did_resume(Hold::NONE)),
                DeviceDeath::ResumedRootFifoMissing,
            );
        }

        /// With maxFIFOCount 0 the child walk leaves the root page and dies
        #[test]
        fn zero_fifo_count_walks_off_page() {
            let t = device(Opts {
                max_fifo_count: 0,
                ..Opts::default()
            });
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            died(
                catch_death(|| t.device.did_resume(Hold::NONE)),
                DeviceDeath::ResumeWalkLeftRootPage,
            );
            assert_eq!(
                DeviceDeath::ResumeWalkLeftRootPage.site().kind,
                DeathKind::OutOfBounds
            );
        }

        /// A display page outside guest RAM raises inside the per-port handler: logged, and the
        /// resume goes on
        #[test]
        fn a_display_raise_is_logged_and_the_resume_goes_on() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            t.set_guest_root_word(ROOT_DISPLAY_WORDS + 1, 0x1000);
            let lines = Lines::capture();
            t.device.did_resume(Hold::NONE);
            // The line carries the raise's detail.
            let caught: Vec<String> = lines
                .all()
                .into_iter()
                .filter(|l| l.starts_with("resume: display port 1 not set up: "))
                .collect();
            assert_eq!(caught.len(), 1, "{caught:?}");
            assert!(caught[0].contains("4096") || caught[0].contains("0x1000"));
            assert!(t.device.displays.lock().guest_display[1].is_none());
            assert_eq!(t.device.run_state.load(), RunState::Running);
        }

        /// The host's `readMemory` block raising under the per-task handler ends the process
        /// at the read
        #[test]
        fn a_raising_read_block_dies_under_the_per_task_catch() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            t.task_root_node(0x20, 0x4000);
            t.set_guest_root_word(2, 0x20);
            t.platform.script().push(
                "readMemory",
                crate::testkit::Step::raise("HostReadRefused", "the host refused the read"),
            );
            let lines = Lines::capture();
            let d = catch_death(|| t.device.did_resume(Hold::NONE)).unwrap_err();
            assert_eq!((d.kind, d.file), (DeathKind::HostRaised, "readMemory"));
            assert!(!lines.contains("not set up"));
        }

        /// The host's `mapMemory` block raising for a display's shared-state page, under the
        /// per-port handler, ends the process at the map
        #[test]
        fn a_raising_map_block_for_a_display_page_dies() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            t.set_guest_root_word(ROOT_DISPLAY_WORDS + 1, 0x10);
            // The root node's mapping passes; the display page's raises.
            t.platform.script().after(
                "mapMemory",
                1,
                crate::testkit::Step::raise("HostMapRefused", "the host refused the map"),
            );
            let lines = Lines::capture();
            let d = catch_death(|| t.device.did_resume(Hold::NONE)).unwrap_err();
            assert_eq!((d.kind, d.file), (DeathKind::HostRaised, "mapMemory"));
            assert!(!lines.contains("not set up"));
        }

        /// With EFI enabled in the registers the EFI display comes back
        #[test]
        fn efi_restored() {
            let t = dev();
            t.device.displays.lock().efi.enabled = 1;
            t.device.did_resume(Hold::NONE);
            assert!(t.device.displays.lock().efi_display.is_some());
            assert_eq!(t.device.displays.lock().efi.enabled, 1);
        }
    }

    /// Writes child node `id` in the root node (guest page [`ROOT_PAGE`]): its stamp index and
    /// running marker (the u16s at +0xc and +0xe, the marker a resume tests at 0x40e) and
    /// its page-list page (+0x10).
    fn child_node(t: &TestDevice, id: u32, stamp: u16, running: u16, page_list: u32) {
        let word = u64::from((0x400 + (id - 1) * 0x14) / 4);
        t.set_guest_root_word(word + 3, u32::from(stamp) | (u32::from(running) << 16));
        t.set_guest_root_word(word + 4, page_list);
    }

    /// Writes guest RAM word `i` of page `page`.
    fn set_page_word(t: &TestDevice, page: u32, i: u64, v: u32) {
        t.platform.fake_vm().sync();
        assert!(
            t.host
                .ram()
                .write_u32(u64::from(page) * APV_PAGE_SIZE + i * 4, v)
        );
        t.platform.fake_vm().sync();
    }

    /// A page list at `list` naming one ring page, `ring`: a child that starts.
    fn page_list(t: &TestDevice, list: u32, ring: u32) {
        set_page_word(t, list, 0, ring);
        set_page_word(t, list, 1, 0);
    }

    /// The registers `will_resume` stores for a 1-page root FIFO at page 2, start 0x400,
    /// enabled, with `fault_offset`; the root page [`ROOT_PAGE`].
    fn resume_registers(t: &TestDevice, fault_offset: u32) {
        use crate::device::test_support::ROOT_PAGE;
        t.device.root_task.lock().root_page_number = ROOT_PAGE;
        let mut r = t.device.root_fifo.lock();
        r.fifo.enabled = 1;
        r.fifo.base_page = 2;
        r.fifo.length = 0x4000;
        r.fifo.start = 0x400;
        r.fifo.read = 0;
        r.fifo.written = 0;
        r.fifo.fault_offset = fault_offset;
    }

    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time polling
    mod suspend_wakeups {
        use super::*;
        use std::time::Duration;

        /// Whether a thread waiting on the signal condition (as a FIFO's stamp wait does) is
        /// woken by `call`, within 300 ms.
        fn wakes(t: &TestDevice, call: impl FnOnce()) -> bool {
            let d = t.device.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            let waiter = std::thread::spawn(move || {
                let mut g = d.signal_mutex.lock();
                tx.send(()).unwrap();
                !d.signal_cond
                    .wait_for(&mut g, Duration::from_millis(300))
                    .timed_out()
            });
            rx.recv().unwrap();
            // The waiter holds the signal mutex until it waits: once taken here, it is waiting.
            drop(t.device.signal_mutex.lock());
            call();
            waiter.join().unwrap()
        }

        /// willSuspend does not signal the FIFOs: a stamp waiter stays blocked; pause (the
        /// control) wakes it
        #[test]
        fn will_suspend_leaves_waiters_blocked() {
            let t = dev();
            assert!(!wakes(&t, || t.device.will_suspend()));
            let t = dev();
            assert!(wakes(&t, || t.device.pause()));
        }
    }

    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time polling
    mod unlocked_reads {
        use super::*;
        use std::time::Duration;

        /// Whether `f` returns within 2 s on another thread.
        fn returns(f: impl FnOnce() + Send + 'static) -> bool {
            let h = std::thread::spawn(f);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !h.is_finished() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            h.is_finished()
        }

        /// willSuspend reads the root FIFO without its lock: a holder (a stamp wait report
        /// mid-describe) does not block it
        #[test]
        fn will_suspend_does_not_wait_on_root_fifo_mutex() {
            let t = dev();
            let d = t.device.clone();
            let held = t.device.root_fifo.lock();
            assert!(returns(move || d.will_suspend()));
            drop(held);
        }

        /// unpause's job on the FIFO queue reads the root FIFO without its lock
        #[test]
        fn unpause_does_not_wait_on_root_fifo_mutex() {
            let t = dev();
            let d = t.device.clone();
            let held = t.device.root_fifo.lock();
            assert!(returns(move || d.unpause()));
            drop(held);
        }

        /// The guest display's client is read without the guest display's lock: the display lookup
        /// by port returns while the guest display's lock is held
        #[test]
        fn display_at_port_does_not_wait_on_guest_display_mutex() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            let guest_display = t.device.get_guest_display_at_port(0).unwrap();
            let d = t.device.clone();
            let held = guest_display.guest_display_mutex.lock();
            assert!(returns(move || assert!(d.get_display_at_port(0).is_none())));
            drop(held);
        }
    }

    mod did_resume_fifos {
        use super::*;
        use crate::device::test_support::ROOT_PAGE;

        /// A restored fault offset sets the root FIFO's register and mask bit 0 only: no fault
        /// stamp, no extra interrupt
        #[test]
        fn restored_fault_offset() {
            let t = dev();
            resume_registers(&t, 0x44);
            t.device.did_resume(Hold::NONE);
            let root = t.device.root_fifo.atomics().root_fifo().expect("root fifo");
            assert_eq!(root.fault_offset(), 0x44);
            assert_eq!(t.device.interrupt_fault_mask.load(Ordering::SeqCst), 1);
            assert_eq!(t.device.interrupt_stamp.load(Ordering::SeqCst), 0);
            assert_eq!(t.host.interrupts(), vec![0]);
            t.device.stop(Hold::NONE);
        }

        /// No fault offset (u32::MAX): the register stays unset and the mask 0
        #[test]
        fn no_fault_offset() {
            let t = dev();
            resume_registers(&t, u32::MAX);
            t.device.did_resume(Hold::NONE);
            let root = t.device.root_fifo.atomics().root_fifo().expect("root fifo");
            assert_eq!(root.fault_offset(), u32::MAX);
            assert_eq!(t.device.interrupt_fault_mask.load(Ordering::SeqCst), 0);
            t.device.stop(Hold::NONE);
        }

        /// Resuming in place: the root page's store tears the live root down, which clears the
        /// root FIFO's `enabled`, so the root FIFO is not recreated
        #[test]
        fn in_place_loses_root_fifo() {
            let t = dev().with_root_page();
            t.device.set_fifo_base_page(2);
            t.device.set_fifo_length(0x4000);
            t.device.set_fifo_start(0x400);
            t.device.enable_fifo();
            assert!(t.device.root_fifo.atomics().root_fifo().is_some());
            t.device.did_resume(Hold::NONE);
            assert!(t.device.root_fifo.atomics().root_fifo().is_none());
            assert_eq!(t.device.root_fifo.atomics().enabled(), 0);
            assert_eq!(t.device.run_state.load(), RunState::Running);
        }

        /// Children come back on the calling thread (the FIFO queue blocked throughout), and only
        /// below `maxFIFOCount`: a marked child 3 with maxFIFOCount 3 is not recreated
        #[test]
        fn children_off_queue_below_max() {
            let t = device(Opts {
                max_fifo_count: 3,
                ..Opts::default()
            });
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            child_node(&t, 1, 1, 1, 0x30);
            child_node(&t, 3, 3, 1, 0x30);
            page_list(&t, 0x30, 0x31);
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            t.device.fifo_queue.exec_async(move || {
                let _ = rx.recv();
            });
            let lines = Lines::capture();
            t.device.did_resume(Hold::NONE);
            {
                let children = t.device.child_fifos.lock();
                assert!(children[1].is_some());
                assert!(children[3].is_none());
            }
            tx.send(()).unwrap();
            assert!(!lines.contains("not created again"));
            assert!(!lines.contains("did not start"));
            t.device.stop(Hold::NONE);
        }

        /// A child restored while the blob's root FIFO is disabled runs with the stamp memory
        /// NULL: its first stamp signal (as its command completion sends it) is the host death
        #[test]
        fn child_without_root_fifo_signals_through_null() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            child_node(&t, 1, 1, 1, 0x30);
            page_list(&t, 0x30, 0x31);
            t.device.did_resume(Hold::NONE);
            assert!(t.device.child_fifos.lock()[1].is_some());
            assert!(t.device.stamp_memory.lock().is_none());
            let d = t.device.clone();
            // The stamp store through the NULL stamp memory: the device's death.
            assert_eq!(
                catch_death(|| d.signal_stamp(1, 1)).expect_err("death"),
                crate::device::DeviceDeath::StampWriteWithoutMemory.site()
            );
            t.device.stop(Hold::NONE);
        }

        /// A marked child that cannot be created is logged; a live child with a fault offset sets
        /// the mask bit of its id, not of its stamp index; a child whose page list is empty logs
        /// that it did not start
        #[test]
        fn child_refused_fault_bits_and_start_failure() {
            let t = dev().with_root_page();
            // child 2: stamp index 5, created before the resume, faulted; its page list is
            // page 0 (empty), so it cannot start.
            child_node(&t, 2, 5, 1, 0);
            assert!(t.device.create_fifo_internal(2, true));
            t.device.child_fifos.lock()[2]
                .as_ref()
                .unwrap()
                .set_fault_offset(0x20);
            // Resuming in place: the root FIFO's `enabled` is 0, so the root teardown leaves child
            // 2, and the marked node makes the resume's create refuse to create it again.
            let lines = Lines::capture();
            t.device.did_resume(Hold::NONE);
            assert_eq!(t.device.interrupt_fault_mask.load(Ordering::SeqCst), 1 << 2);
            assert_eq!(lines.count("child FIFO 2 refused: it exists"), 1);
            assert_eq!(lines.count("resume: child FIFO 2 not created again"), 1);
            assert_eq!(lines.count("child FIFO 2 did not start"), 1);
        }

        /// Unpause's job starts the same child, and logs its refusal once
        #[test]
        fn unpause_logs_the_start_failure() {
            let t = dev().with_root_page();
            // child 2: page list at page 0 (empty), so it cannot start.
            child_node(&t, 2, 5, 1, 0);
            assert!(t.device.create_fifo_internal(2, true));
            let lines = Lines::capture();
            t.device.unpause();
            assert_eq!(lines.all(), ["child FIFO 2 did not start"]);
        }
    }

    mod did_resume_tasks {
        use super::*;
        use crate::device::test_support::ROOT_PAGE;

        /// A task root outside guest RAM fails the read (logged); the define still runs, with the
        /// zeroed length
        #[test]
        fn unreadable_task_root() {
            let t = dev();
            t.device.root_task.lock().root_page_number = ROOT_PAGE;
            t.set_guest_root_word(4, 0x1000);
            let lines = Lines::capture();
            t.device.did_resume(Hold::NONE);
            assert_eq!(
                lines.count(
                    "resume: task 4's root node at page 0x1000 not readable, so its length reads 0"
                ),
                1
            );
            assert_eq!(lines.count("resume: task 4 not set up: "), 1);
            assert!(t.device.tasks.lock().get(4).is_none());
            assert_eq!(t.device.run_state.load(), RunState::Running);
        }
    }

    mod guest_displays_paused {
        use super::*;

        /// Pause and finishSuspend pause the guest displays, unpause unpauses them
        #[test]
        fn guest_displays_follow() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            let guest_display = t.device.get_guest_display_at_port(0).unwrap();
            t.device.pause();
            assert!(guest_display.guest_display_mutex.lock().paused);
            t.device.unpause();
            assert!(!guest_display.guest_display_mutex.lock().paused);
            t.device.will_suspend();
            assert!(t.device.finish_suspend().is_some());
            assert!(guest_display.guest_display_mutex.lock().paused);
        }

        /// Pause, unpause and finishSuspend read the guest displays without the displays' lock:
        /// with a page 0 or a detach holding it (for up to 2 s), they reach the guest displays at
        /// once
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread
        fn guest_displays_follow_while_display_mutex_is_held() {
            use std::sync::mpsc;
            use std::time::Duration;
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            let guest_display = t.device.get_guest_display_at_port(0).unwrap();
            let held = t.device.displays.lock();
            let run = |what: &'static str, f: fn(&DeviceCore<crate::testkit::FakePlatform>)| {
                let dev = t.device.clone();
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    f(&dev);
                    let _ = tx.send(());
                });
                rx.recv_timeout(Duration::from_secs(3))
                    .unwrap_or_else(|_| panic!("{what} waited on the displays' lock"));
            };
            run("pause", |d| d.pause());
            assert!(guest_display.guest_display_mutex.lock().paused);
            run("unpause", |d| d.unpause());
            assert!(!guest_display.guest_display_mutex.lock().paused);
            run("finish_suspend", |d| {
                d.will_suspend();
                assert!(d.finish_suspend().is_some());
            });
            assert!(guest_display.guest_display_mutex.lock().paused);
            drop(held);
        }
    }

    mod pause {
        use super::*;

        /// The paused state is set last
        #[test]
        fn pauses() {
            let t = dev();
            t.device.pause();
            assert_eq!(t.device.run_state.load(), RunState::Paused);
        }
    }

    mod unpause {
        use super::*;

        /// Running again, then one interrupt
        #[test]
        fn unpauses() {
            let t = dev();
            t.device.pause();
            t.device.unpause();
            assert_eq!(t.device.run_state.load(), RunState::Running);
            assert_eq!(t.host.interrupts(), vec![0]);
        }
    }

    mod stop_reset {
        use super::*;

        /// Stop with no root leaves the root page 0
        #[test]
        fn stop() {
            let t = dev();
            t.device.stop(Hold::NONE);
            assert_eq!(t.device.root_task.lock().root_page_number, 0);
        }

        /// Reset: version 0, EFI off, max key 1
        #[test]
        fn reset() {
            let t = dev();
            t.device
                .guest_device_info_max_key
                .store(40, Ordering::Relaxed);
            t.device.displays.lock().efi.start = 9;
            t.device.reset(Hold::NONE);
            assert_eq!(
                t.device.guest_device_info_max_key.load(Ordering::Relaxed),
                1
            );
            assert_eq!(t.device.binary_version.load(Ordering::Relaxed), 0);
            assert_eq!(t.device.displays.lock().efi.start, 0);
        }
    }
}
