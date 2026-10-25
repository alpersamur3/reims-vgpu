//! The root: the root node, the root task and its address space, the root FIFO and its
//! registers, and the device's mappings.
//!
//! Without a memory map the root task is the host's 16 MiB task (`createTask`), and every
//! device mapping is a range of it: the range comes from the root task's [`FirstFitVa`], which
//! is never reset, and the host's `mapMemory` maps it. With a memory map the mappings come
//! from the map and there is no root task.
//!
//! ## Contract
//!
//! - A new root page tears the live root down (FIFOs, tasks, displays, the root node, the
//!   interrupt state, then the root task), idles the waiters, then makes the root task and maps
//!   the root node under the root node's lock; the page is stored even when the mapping fails
//!   (`tests::set_root_page_blocks_order`, `tests::set_root_page_twice_replaces`).
//! - A register read from inside the host's `mapMemory` or `createTask` block answers from the
//!   staged view without waiting on the locks the root page holds around the call
//!   (`mmio::tests::map_memory_block_reading_root_page_register_returns`).
//! - The root FIFO's teardown stops and drains the root FIFO, deletes the children, then under
//!   the root FIFO's lock clears the stamp memory, releases the FIFO and clears the enable, and
//!   publishes those stores in one step before it releases the mapping, whose release calls
//!   the host (`tests::teardown_fifos_publishes_before_the_unmap`).
//! - A failed root-node mapping destroys the root task but leaves the task handle set; only the
//!   base is cleared (`tests::failed_root_map_leaves_task_dangling`).
//! - A NULL task from `createTask` is not checked: the mapping fails and `destroyTask(NULL)` is
//!   called (`tests::null_create_task_destroys_null`).
//! - A mapping released after its root task skips the host unmap and the allocator's free: the
//!   range leaks for the device's life (`tests::mapping_outliving_root_task_leaks`).
//! - The root FIFO's teardown skips children at or past the current `maxFIFOCount`
//!   (`tests::teardown_fifos_keeps_children_past_max`).
//! - The root FIFO's drain waits out a `CmdDelay`, so the MMIO write that tears it down hangs as
//!   long.
//!
//! [`FirstFitVa`]: crate::collections::FirstFitVa

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::device::{DeviceCore, DeviceDeath, DeviceRaise, MappingSink, ROOT_TASK_SIZE};
use crate::fifo::Fifo;
use crate::guest::GuestPtr;
use crate::guest::records::{APV_PAGE_MASK, APV_PAGE_SHIFT, APV_PAGE_SIZE};
use crate::host::{HostPtr, PhysRange};
use crate::memory::mapping::{Mapping, Teardown};
use crate::num::{ToU64, ToUsize, Truncate};
use crate::platform::Platform;
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::rt::Hold;
use crate::seams::links::FifoDeviceLink;

/// The root FIFO's largest length.
pub const ROOT_FIFO_MAX_LENGTH: u32 = 0x2_0000;
/// The root FIFO's start alignment mask (a multiple of 0x400).
pub const ROOT_FIFO_START_ALIGN_MASK: u32 = 0x3ff;
/// A healthy FIFO's fault offset.
pub const NOT_FAULTED: u32 = u32::MAX;
/// The most a physical read reads: one guest page.
pub const PHYSICAL_READ_PAGE: u64 = APV_PAGE_SIZE;
/// A physical address's offset in its guest page.
const PHYSICAL_PAGE_MASK: u64 = APV_PAGE_MASK;

/// A call whose raise no frame above it catches: the process dies at the raise's site, under a
/// handler or not.
pub(crate) fn or_die<T, E: Into<Raise>>(r: Result<T, E>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => e.into().fatal(),
    }
}

/// A page number's guest-physical address: a 32-bit page reaches 2^46 bytes (2^44 on 4 KiB guest
/// pages).
fn page_address(page: u32) -> u64 {
    u64::from(page).wrapping_mul(APV_PAGE_SIZE)
}

impl<P: Platform> DeviceCore<P> {
    /// Tears the live root down, idles the waiters, then (for a non-zero page) makes the root
    /// task without a memory map and maps the root node. The page number is stored even when
    /// the mapping fails.
    ///
    /// Its callers have no handler (the host's MMIO write, a resume, a stop or reset, and the
    /// device's shutdown), and the host's blocks are called where no handler catches: a raise
    /// out of one ends the process.
    ///
    /// A failed root-node mapping destroys the root task but leaves the task handle set (only
    /// the base is cleared). A NULL task from `createTask` is not checked: the mapping then fails
    /// and `destroyTask(NULL)` is called.
    pub fn set_root_page(&self, hold: Hold<'_>, root_page: u32) {
        if self.root_node.atomics().root_node().is_some() {
            self.teardown_root(hold);
        }
        self.init_waiting_info();
        if root_page != 0 {
            let mut node = self.root_node.lock();
            let has_memory_map = self.memory_map.lock().is_some();
            if !has_memory_map {
                let Some(create) = &self.blocks.create_task else {
                    crate::fatal!(DeviceDeath::NilBlock("createTask"))
                };
                // The task and its base are stored under the root node's lock only, not the
                // root task's. The block writes the base in place, over the old one.
                let rt = self.root_task.atomics();
                let mut base = rt.task_info().base_address;
                let task = create(ROOT_TASK_SIZE, &mut base);
                rt.root_task_base.store(base.addr(), Ordering::Relaxed);
                rt.root_task.store(task.addr(), Ordering::Relaxed);
            }
            let mapping = self.new_contiguous_mapping(root_page, APV_PAGE_SIZE);
            match mapping {
                Some(mapping) => {
                    node.root_node = Some(GuestPtr::new(mapping.virtual_address()));
                    node.root_node_mapping = Some(mapping);
                }
                None => {
                    node.root_node_mapping = None;
                    log::warn!("root page {root_page:#x} not mapped: no root node");
                    if !has_memory_map {
                        let task = self.root_task.atomics().task_info().task;
                        let Some(destroy) = &self.blocks.destroy_task else {
                            crate::fatal!(DeviceDeath::NilBlock("destroyTask"))
                        };
                        destroy(task);
                    }
                    node.root_node = None;
                    // The base is cleared; the task handle is kept.
                    self.root_task
                        .atomics()
                        .root_task_base
                        .store(0, Ordering::Relaxed);
                }
            }
        }
        self.root_task
            .atomics()
            .root_page_number
            .store(root_page, Ordering::Relaxed);
    }

    /// FIFOs, tasks, displays, the root node, the interrupt state, then the root task. Its
    /// allocator and base are kept.
    pub fn teardown_root(&self, hold: Hold<'_>) {
        crate::fatal_unless!(
            self.root_node.atomics().root_node().is_some(),
            DeviceDeath::TeardownWithoutRootNode
        );
        self.tearing_down_root.store(true, Ordering::Relaxed);
        self.teardown_fifos();
        self.teardown_tasks();
        self.teardown_displays(hold);
        self.root_task
            .atomics()
            .read_page
            .store(0, Ordering::Relaxed);
        // The root node is cleared in its staged view first; the mapping's release (the host's
        // `unmapMemory` and the allocator's free) runs with the root node's lock free, which is
        // taken only to take the mapping out.
        let root_node = &self.root_node.atomics().root_node;
        if root_node.load(Ordering::Relaxed) != 0 {
            root_node.store(0, Ordering::Relaxed);
            let mapping = self.root_node.lock().root_node_mapping.take();
            drop(mapping);
        }
        self.interrupt_fault_mask.store(0, Ordering::SeqCst);
        self.interrupt_stamp.store(0, Ordering::SeqCst);
        self.interrupt_display.store(0, Ordering::SeqCst);
        {
            let _rt = self.root_task.lock();
            let info = self.root_task.atomics().task_info();
            if !info.task.is_null() {
                let Some(destroy) = &self.blocks.destroy_task else {
                    crate::fatal!(DeviceDeath::TeardownWithoutDestroyTask)
                };
                destroy(info.task);
                self.root_task
                    .atomics()
                    .root_task
                    .store(0, Ordering::Relaxed);
            }
        }
        self.tearing_down_root.store(false, Ordering::Relaxed);
    }

    /// Stops and drains the root FIFO, deletes the children, then releases the root FIFO and
    /// its mapping under the root FIFO's lock. Hangs where the drain does (a `CmdDelay`).
    ///
    /// A child delete refuses ids at or past the current `maxFIFOCount`, so such children survive
    /// the teardown, still running, while the stamp memory goes.
    ///
    /// Its callers are the host's MMIO write, a resume, a stop or reset (each an entry's), and the
    /// device's shutdown (a drop's). None has a handler, so a child delete's raise dies at the
    /// raise, and none reaches `or_die`.
    pub fn teardown_fifos(&self) {
        if self.root_fifo.atomics().enabled() == 0 {
            return;
        }
        let root = self.root_fifo.atomics().root_fifo();
        if let Some(fifo) = &root {
            fifo.stop();
        }
        self.signal_fifos();
        if let Some(fifo) = &root {
            fifo.drain();
        }
        drop(root);
        let slot0_empty = self
            .child_fifos
            .lock()
            .first()
            .is_none_or(|slot| slot.is_none());
        crate::fatal_unless!(slot0_empty, DeviceDeath::ChildInSlotZero);
        // Each slot is read as the loop reaches it.
        for id in 1u32..64 {
            let present = self
                .child_fifos
                .lock()
                .get(id.to_usize())
                .is_some_and(Option::is_some);
            if present {
                or_die(self.delete_fifo(id));
            }
        }
        {
            let mut g = self.root_fifo.lock();
            *self.stamp_memory.lock() = None;
            drop(g.root_fifo.take());
            g.fifo.enabled = 0;
            // Seen by unlocked readers before the release below calls the host.
            g.publish();
            drop(g.root_fifo_mapping.take());
        }
        self.interrupt_fault_mask.store(0, Ordering::SeqCst);
    }

    /// Validates the staged registers, maps `[basePage·APV_PAGE_SIZE, +length)`, and makes the root
    /// FIFO; every refusal only logs (0x1000 keeps reading 0).
    ///
    /// Its callers have no handler (the host's MMIO write and a resume): a raise making the
    /// FIFO's shared event dies at the send.
    pub fn create_root_fifo(&self) {
        {
            let node = self.root_node.lock();
            if node.root_node.is_none() {
                log::warn!("root FIFO not created: no root node");
                return;
            }
        }
        let mut g = self.root_fifo.lock();
        let length = g.fifo.length;
        if length > ROOT_FIFO_MAX_LENGTH {
            log::warn!(
                "root FIFO not created: its length {length:#x} is above {ROOT_FIFO_MAX_LENGTH:#x}"
            );
            return;
        }
        // Length 0 passes.
        if u64::from(length) % const { NonZeroU64::new(APV_PAGE_SIZE).unwrap() } != 0 {
            log::warn!("root FIFO not created: its length {length:#x} is not whole pages");
            return;
        }
        if g.fifo.start == 0 || g.fifo.start & ROOT_FIFO_START_ALIGN_MASK != 0 {
            log::warn!(
                "root FIFO not created: its start {:#x} is 0 or not a multiple of {:#x}",
                g.fifo.start,
                ROOT_FIFO_START_ALIGN_MASK + 1
            );
            return;
        }
        // Under the root FIFO's lock (root FIFO, then root task). No handler catches here.
        g.root_fifo_mapping = self.new_contiguous_mapping(g.fifo.base_page, u64::from(length));
        let Some(base) = g.root_fifo_mapping.as_ref().map(|m| m.virtual_address()) else {
            log::warn!(
                "root FIFO not created: its {length:#x} bytes at page {:#x} are not mapped",
                g.fifo.base_page
            );
            return;
        };
        let host: FifoDeviceLink<P> = self.weak_self.clone();
        let last_pending = self
            .waiting_info
            .first()
            .map(|w| w.last_pending_stamp.clone())
            .unwrap_or_default();
        let Some(fifo) = Fifo::try_new_root_on(
            &self.platform,
            host,
            g.fifo.length,
            GuestPtr::new(base),
            g.fifo.start,
            last_pending,
        ) else {
            // The FIFO's own line says why.
            log::warn!("root FIFO not created: the FIFO refused its registers");
            drop(g.root_fifo_mapping.take());
            return;
        };
        g.root_fifo = Some(fifo);
        *self.stamp_memory.lock() = Some(GuestPtr::new(base));
        g.fifo.enabled = 1;
    }

    /// Creates the root FIFO and starts it; the start's result is ignored.
    pub fn enable_fifo(&self) {
        crate::fatal_unless!(
            self.root_fifo.atomics().enabled() == 0,
            DeviceDeath::FifoAlreadyEnabled
        );
        self.create_root_fifo();
        if self.root_fifo.atomics().enabled() != 0
            && let Some(fifo) = self.root_fifo.atomics().root_fifo()
        {
            let _ = fifo.start(&self.platform);
        }
    }

    /// The root FIFO's doorbell. The FIFO's lock nests inside the root FIFO's.
    ///
    /// With no root FIFO the doorbell is dropped (not staged).
    pub fn set_fifo_written(&self, written: u32) {
        let g = self.root_fifo.lock();
        if let Some(fifo) = &g.root_fifo {
            fifo.set_root_written(written);
        }
    }

    /// Stages the root FIFO's length; refused, with a log, while it is enabled.
    pub fn set_fifo_length(&self, length: u32) {
        let mut g = self.root_fifo.lock();
        if g.fifo.enabled != 0 {
            log::warn!("root FIFO length write of {length:#x} dropped: the FIFO is enabled");
            return;
        }
        g.fifo.length = length;
    }

    /// Stages the root FIFO's start offset; refused, with a log, while it is enabled.
    pub fn set_fifo_start(&self, start: u32) {
        let mut g = self.root_fifo.lock();
        if g.fifo.enabled != 0 {
            log::warn!("root FIFO start write of {start:#x} dropped: the FIFO is enabled");
            return;
        }
        g.fifo.start = start;
    }

    /// Stages the root FIFO's base page; refused, with a log, while it is enabled.
    pub fn set_fifo_base_page(&self, page: u32) {
        let mut g = self.root_fifo.lock();
        if g.fifo.enabled != 0 {
            log::warn!("root FIFO base page write of {page:#x} dropped: the FIFO is enabled");
            return;
        }
        g.fifo.base_page = page;
    }

    /// The root FIFO's fault offset.
    ///
    /// 0 with no root FIFO.
    pub fn fifo_fault_offset(&self) -> u32 {
        let g = self.root_fifo.lock();
        g.root_fifo.as_ref().map_or(0, |f| f.fault_offset())
    }

    /// On the FIFO queue: clears the root's fault bit (when the root FIFO is faulted, or
    /// absent), raises the interrupt for the faults left, then resumes the root FIFO. The
    /// signal lock is held across the interrupt.
    pub fn resume_fifo(&self) {
        self.on_fifo_queue(|d| {
            let root = d.root_fifo.atomics().root_fifo();
            {
                let _signal = d.signal_mutex.lock();
                // No root FIFO answers 0, which is "faulted".
                if root.as_ref().map_or(0, |f| f.fault_offset()) != NOT_FAULTED {
                    d.interrupt_fault_mask.fetch_and(!1, Ordering::SeqCst);
                    if d.interrupt_fault_mask.load(Ordering::SeqCst) != 0 {
                        d.interrupt();
                    }
                }
            }
            drop(root);
            // Read again.
            if let Some(fifo) = d.root_fifo.atomics().root_fifo() {
                fifo.resume();
            }
        });
    }

    /// A dying root-task mapping's unmap ([`MappingSink::unmap_range`]).
    pub fn unmap_range(&self, virtual_address: HostPtr, length: u64) {
        self.mapping_sink.unmap_range(virtual_address, length);
    }

    /// `[page·APV_PAGE_SIZE, +length)` read-write: from the memory map, or a range of the root task
    /// the host maps. `None` when it cannot be made.
    pub fn new_contiguous_mapping(&self, page: u32, length: u64) -> Option<Arc<Mapping<P>>> {
        crate::fatal_unless!(
            !self.leaving_suspend_early(),
            DeviceDeath::MappingWhileLeavingSuspend
        );
        let range = PhysRange {
            physical_address: page_address(page),
            physical_length: length,
        };
        let map = self.memory_map.lock().clone();
        if let Some(map) = map {
            return map.new_range_mapping(&[range], length, 0);
        }
        self.root_task_mapping(&[range], length)
    }

    /// The ranges mapped read-write, one after another: from the memory map, or a range of the
    /// root task the host maps. `read_only` is not passed on. `None` when it cannot be made.
    pub fn new_range_mapping(
        &self,
        ranges: &[PhysRange],
        total_length: u64,
        read_only: bool,
    ) -> Option<Arc<Mapping<P>>> {
        let _ = read_only;
        crate::fatal_unless!(
            !self.leaving_suspend_early(),
            DeviceDeath::MappingWhileLeavingSuspend
        );
        crate::fatal_unless!(
            total_length % const { NonZeroU64::new(APV_PAGE_SIZE).unwrap() } == 0,
            DeviceDeath::UnalignedRangeMapping
        );
        let map = self.memory_map.lock().clone();
        if let Some(map) = map {
            return map.new_range_mapping(ranges, total_length, 0);
        }
        self.root_task_mapping(ranges, total_length)
    }

    /// The half of the two mapping methods without a memory map: a range of the root task's
    /// address space, mapped by the host, made under the root task's lock. A raise out of the
    /// host's block leaves the range allocated.
    fn root_task_mapping(&self, ranges: &[PhysRange], length: u64) -> Option<Arc<Mapping<P>>> {
        let mut rt = self.root_task.lock();
        let info = self.root_task.atomics().task_info();
        if info.task.is_null() {
            return None;
        }
        let offset = rt
            .root_task_allocator
            .as_mut()
            .and_then(|a| a.allocate(length))?;
        let virtual_address = info.base_address.wrapping_add(offset);
        let Some(map) = &self.blocks.map_memory else {
            crate::fatal!(DeviceDeath::NilBlock("mapMemory"))
        };
        if !map(info.task, ranges.len().low_u32(), offset, false, ranges) {
            if let Some(allocator) = rt.root_task_allocator.as_mut() {
                allocator.deallocate(offset, length);
            }
            return None;
        }
        // The mapping holds the device through the sink, which answers until the device's
        // fields are gone.
        Some(Mapping::new(
            virtual_address,
            length,
            true,
            Teardown::RootTask {
                device: self.mapping_link(),
            },
        ))
    }

    /// Reads `dst.len()` bytes (a 32-bit length) of one guest page: through the memory map,
    /// the host's `readMemory` block, or a root-task mapping of the page. Raises for a read
    /// past the page's end, as `on` says: its callers reach it both under a handler and with
    /// none.
    pub fn read_from_physical_address(
        &self,
        on: OnFault,
        address: u64,
        dst: &mut [u8],
    ) -> RaiseResult<bool> {
        let length = dst.len().low_u32();
        if u64::from(length) > PHYSICAL_READ_PAGE
            || (address & PHYSICAL_PAGE_MASK).wrapping_add(u64::from(length)) > PHYSICAL_READ_PAGE
        {
            crate::raise!(on => DeviceRaise::ReadPastPage,
                format_args!("read of {length:#x} bytes at {address:#x} runs past its page")
            );
        }
        let map = self.memory_map.lock().clone();
        if let Some(map) = map {
            return Ok(map.read(address, dst));
        }
        if let Some(read) = &self.blocks.read_memory {
            let destination = HostPtr::from_ptr(dst.as_mut_ptr());
            return Ok(read(address, u64::from(length), destination));
        }
        // The page number truncated to 32 bits.
        let page = address.wrapping_shr(APV_PAGE_SHIFT).low_u32();
        let Some(mapping) = self.new_contiguous_mapping(page, PHYSICAL_READ_PAGE) else {
            return Ok(false);
        };
        GuestPtr::new(mapping.virtual_address()).copy_to(
            address & PHYSICAL_PAGE_MASK,
            dst,
            DeviceDeath::PhysicalReadCopy,
        );
        drop(mapping);
        Ok(true)
    }
}

impl<P: Platform> MappingSink<P> {
    /// A dying root-task mapping's unmap, over its address and length: the host's
    /// `unmapMemory` and the allocator's free, under the root task's lock. The base is read and
    /// the offset checked before the lock is taken.
    ///
    /// With no root task the host unmap and the allocator free are both skipped: the range leaks
    /// for the device's life.
    pub fn unmap_range(&self, virtual_address: HostPtr, length: u64) {
        if self.memory_map.lock().is_some() {
            return;
        }
        crate::fatal_unless!(
            !self.leaving_suspend_early(),
            DeviceDeath::MappingWhileLeavingSuspend
        );
        let base = self.root_task.atomics().task_info().base_address;
        let offset = virtual_address
            .addr()
            .to_u64()
            .wrapping_sub(base.addr().to_u64());
        crate::fatal_unless!(offset < ROOT_TASK_SIZE, DeviceDeath::UnmapOutsideRootTask);
        let mut rt = self.root_task.lock();
        let task = self.root_task.atomics().task_info().task;
        if !task.is_null() {
            let Some(unmap) = &self.blocks.unmap_memory else {
                crate::fatal!(DeviceDeath::NilBlock("unmapMemory"))
            };
            // Its result is ignored.
            let _ = unmap(task, offset, length);
            if let Some(allocator) = rt.root_task_allocator.as_mut() {
                allocator.deallocate(offset, length);
            }
        }
    }
}

/// A device on the fake platform over 4 MiB of fake guest RAM at guest-physical 0, with the
/// fake host's blocks: the setup the root task and root-task mappings run in. Shared by the
/// device's unit tests.
#[cfg(test)]
pub(crate) mod rig {
    use std::sync::Arc;

    use crate::device::DeviceCore;
    use crate::device::config::DeviceConfig;
    use crate::host::HostBlocks;
    use crate::rt::Hold;
    use crate::testkit::calllog::Call;
    use crate::testkit::host::FakeHost;
    use crate::testkit::{FakePlatform, GuestRam};

    /// Guest RAM size (pages 0-255).
    pub(crate) const RAM: u64 = 0x40_0000;

    /// The device under test and its fakes.
    pub(crate) struct Rig {
        pub platform: Arc<FakePlatform>,
        pub host: Arc<FakeHost>,
        pub device: Arc<crate::device::Handle<FakePlatform>>,
    }

    /// A rig with every block and `maxFIFOCount` 64.
    pub(crate) fn rig() -> Rig {
        rig_with(|_| {})
    }

    /// A rig whose blocks `edit` changes first (e.g. drops one).
    pub(crate) fn rig_with(edit: impl FnOnce(&mut HostBlocks)) -> Rig {
        let platform = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&platform, GuestRam::with_range(0, RAM).expect("ram"));
        let mut c = DeviceConfig::new(Some(platform.device()));
        c.display_port_count = 1;
        c.blocks = host.blocks();
        edit(&mut c.blocks);
        let device = DeviceCore::create(platform.clone(), c).expect("device");
        let mut f = device.features.load();
        f.max_fifo_count = 64;
        device.features.store(&f);
        Rig {
            platform,
            host,
            device,
        }
    }

    impl Rig {
        /// The host block calls since `mark`.
        pub(crate) fn host_calls(&self, mark: u64) -> Vec<Call> {
            self.platform
                .call_log()
                .since(mark)
                .into_iter()
                .filter(|c| c.class == "HostBlocks")
                .collect()
        }

        /// The host block selectors since `mark`.
        pub(crate) fn host_sels(&self, mark: u64) -> Vec<&'static str> {
            self.host_calls(mark).into_iter().map(|c| c.sel).collect()
        }

        /// The call log's mark now.
        pub(crate) fn mark(&self) -> u64 {
            self.platform.call_log().mark()
        }

        /// Guest RAM word at guest-physical `pa`.
        pub(crate) fn ram_u32(&self, pa: u64) -> u32 {
            let mut b = [0u8; 4];
            assert!(self.host.ram().read(pa, &mut b));
            u32::from_le_bytes(b)
        }

        /// Writes the guest RAM word at guest-physical `pa`.
        pub(crate) fn set_ram_u32(&self, pa: u64, v: u32) {
            assert!(self.host.ram().write(pa, &v.to_le_bytes()));
        }

        /// Sets the root page and a 1-page root FIFO at page `fifo_page`, start 0x400, and
        /// enables it.
        pub(crate) fn with_root_fifo(&self, root_page: u32, fifo_page: u32) {
            self.device.set_root_page(Hold::NONE, root_page);
            self.device.set_fifo_base_page(fifo_page);
            self.device.set_fifo_length(0x4000);
            self.device.set_fifo_start(0x400);
            self.device.enable_fifo();
            assert_eq!(self.device.root_fifo.lock().fifo.enabled, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::rig::{Rig, rig, rig_with};
    use super::*;
    use crate::device::RunState;
    use crate::device::test_support::Lines;
    use crate::raise::RaiseId;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::capture::expect_death;
    use crate::testkit::script::Step;

    /// The root task's handle and base (`None` while both are NULL, as they start).
    fn root_task(r: &Rig) -> Option<crate::tasks::TaskInfo> {
        let info = r.device.root_task.atomics().task_info();
        (!info.task.is_null() || !info.base_address.is_null()).then_some(info)
    }

    #[test]
    fn set_root_page_blocks_order() {
        let r = rig();
        r.set_ram_u32(APV_PAGE_SIZE, 0xabcd);
        let m = r.mark();
        r.device.set_root_page(Hold::NONE, 1);
        let calls = r.host_calls(m);
        let sels: Vec<_> = calls.iter().map(|c| c.sel).collect();
        assert_eq!(sels, ["createTask", "mapMemory"]);
        assert_eq!(calls[0].args[0].as_u64(), Some(ROOT_TASK_SIZE));
        // One range, at allocator offset 0, read-write.
        assert_eq!(calls[1].args[1], Arg::U32(1));
        assert_eq!(calls[1].args[2].as_u64(), Some(0));
        assert!(matches!(calls[1].args[3], Arg::Bool(false)));
        let node = r.device.root_node.lock().root_node.expect("root node");
        let task = root_task(&r).expect("root task");
        assert_eq!(node.addr(), task.base_address);
        assert_eq!(r.device.root_task.lock().root_page_number, 1);
        // The node is guest page 1.
        assert_eq!(node.read_u32(0, DeviceDeath::PhysicalReadCopy), 0xabcd);
    }

    #[test]
    fn set_root_page_zero_tears_down() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let m = r.mark();
        r.device.set_root_page(Hold::NONE, 0);
        assert_eq!(r.host_sels(m), ["unmapMemory", "destroyTask"]);
        assert!(r.device.root_node.lock().root_node.is_none());
        assert!(root_task(&r).expect("info").task.is_null());
        assert_eq!(r.device.root_task.lock().root_page_number, 0);
        assert!(r.host.tasks().is_empty());
    }

    #[test]
    fn set_root_page_twice_replaces() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let m = r.mark();
        r.device.set_root_page(Hold::NONE, 2);
        assert_eq!(
            r.host_sels(m),
            ["unmapMemory", "destroyTask", "createTask", "mapMemory"]
        );
        assert_eq!(r.host.tasks().len(), 1);
    }

    // A failed root-node mapping without a memory map destroys the root task but leaves its
    // handle set.
    #[test]
    fn failed_root_map_leaves_task_dangling() {
        let r = rig();
        r.platform.script().push("mapMemory", Step::value(false));
        let m = r.mark();
        r.device.set_root_page(Hold::NONE, 1);
        assert_eq!(r.host_sels(m), ["createTask", "mapMemory", "destroyTask"]);
        assert!(r.device.root_node.lock().root_node.is_none());
        let info = root_task(&r).expect("info");
        assert!(!info.task.is_null(), "the destroyed task stays set");
        assert!(info.base_address.is_null());
        assert_eq!(r.device.root_task.lock().root_page_number, 1);
    }

    // A root page with a NULL task from `createTask` goes on.
    #[test]
    fn null_create_task_destroys_null() {
        let r = rig();
        r.platform.script().push("createTask", Step::Nil);
        let m = r.mark();
        r.device.set_root_page(Hold::NONE, 1);
        let calls = r.host_calls(m);
        let sels: Vec<_> = calls.iter().map(|c| c.sel).collect();
        assert_eq!(sels, ["createTask", "destroyTask"]);
        assert_eq!(calls[1].args[0], Arg::Ptr(0));
        assert!(r.device.root_node.lock().root_node.is_none());
    }

    #[test]
    fn nil_create_task_block_dies() {
        let r = rig_with(|b| b.create_task = None);
        let d = r.device.clone();
        expect_death(DeviceDeath::NilBlock("createTask").site(), move || {
            d.set_root_page(Hold::NONE, 1)
        });
    }

    #[test]
    fn teardown_root_without_root_node_dies() {
        let r = rig();
        let d = r.device.clone();
        expect_death(DeviceDeath::TeardownWithoutRootNode.site(), move || {
            d.teardown_root(Hold::NONE)
        });
    }

    #[test]
    fn teardown_root_nil_destroy_dies() {
        let r = rig_with(|b| b.destroy_task = None);
        r.device.set_root_page(Hold::NONE, 1);
        let d = r.device.clone();
        expect_death(DeviceDeath::TeardownWithoutDestroyTask.site(), move || {
            d.set_root_page(Hold::NONE, 0)
        });
    }

    #[test]
    fn teardown_root_clears_interrupt_state() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        r.device.interrupt_fault_mask.store(5, Ordering::SeqCst);
        r.device.interrupt_stamp.store(6, Ordering::SeqCst);
        r.device.interrupt_display.store(7, Ordering::SeqCst);
        r.device.teardown_root(Hold::NONE);
        assert_eq!(r.device.interrupt_fault_mask.load(Ordering::SeqCst), 0);
        assert_eq!(r.device.interrupt_stamp.load(Ordering::SeqCst), 0);
        assert_eq!(r.device.interrupt_display.load(Ordering::SeqCst), 0);
        assert!(!r.device.tearing_down_root.load(Ordering::SeqCst));
    }

    /// Each refusal leaves the FIFO off; its line says which check refused
    #[test]
    fn create_root_fifo_refusals() {
        let r = rig();
        let lines = Lines::capture();
        r.device.create_root_fifo();
        r.device.set_root_page(Hold::NONE, 1);
        r.device.set_fifo_length(0x2_4000);
        r.device.create_root_fifo();
        // A quarter of a guest page.
        let quarter = APV_PAGE_SIZE / 4;
        r.device.set_fifo_length(quarter as u32);
        r.device.create_root_fifo();
        r.device.set_fifo_length(0x4000);
        r.device.set_fifo_start(0);
        r.device.create_root_fifo();
        r.device.set_fifo_start(0x401);
        r.device.create_root_fifo();
        assert_eq!(r.device.root_fifo.lock().fifo.enabled, 0);
        // The lines, in order.
        assert_eq!(
            lines.all(),
            [
                "root FIFO not created: no root node".to_owned(),
                "root FIFO not created: its length 0x24000 is above 0x20000".to_owned(),
                // 0x1000 on a 16 KiB guest page.
                format!("root FIFO not created: its length {quarter:#x} is not whole pages"),
                "root FIFO not created: its start 0x0 is 0 or not a multiple of 0x400".to_owned(),
                "root FIFO not created: its start 0x401 is 0 or not a multiple of 0x400".to_owned(),
            ]
        );
    }

    #[test]
    fn create_root_fifo_start_past_length_releases_mapping() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        r.device.set_fifo_base_page(2);
        r.device.set_fifo_length(0x4000);
        r.device.set_fifo_start(0x4000);
        let m = r.mark();
        r.device.create_root_fifo();
        assert_eq!(r.host_sels(m), ["mapMemory", "unmapMemory"]);
        let g = r.device.root_fifo.lock();
        assert!(g.root_fifo.is_none() && g.root_fifo_mapping.is_none());
        assert_eq!(g.fifo.enabled, 0);
    }

    #[test]
    fn create_root_fifo_failed_map() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        r.device.set_fifo_base_page(2);
        r.device.set_fifo_length(0x4000);
        r.device.set_fifo_start(0x400);
        r.platform.script().push("mapMemory", Step::value(false));
        let lines = Lines::capture();
        r.device.create_root_fifo();
        assert_eq!(
            lines.all(),
            ["root FIFO not created: its 0x4000 bytes at page 0x2 are not mapped"]
        );
        assert_eq!(r.device.root_fifo.lock().fifo.enabled, 0);
    }

    #[test]
    fn enable_fifo_and_registers_locked_while_enabled() {
        let r = rig();
        r.with_root_fifo(1, 2);
        assert!(r.device.stamp_memory.lock().is_some());
        r.device.set_fifo_length(0x8000);
        r.device.set_fifo_start(0x800);
        r.device.set_fifo_base_page(9);
        let g = r.device.root_fifo.lock();
        assert_eq!(
            (g.fifo.length, g.fifo.start, g.fifo.base_page),
            (0x4000, 0x400, 2)
        );
        drop(g);
        // A healthy root FIFO's fault offset.
        assert_eq!(r.device.fifo_fault_offset(), NOT_FAULTED);
        r.device.teardown_fifos();
        let g = r.device.root_fifo.lock();
        assert!(g.root_fifo.is_none() && g.root_fifo_mapping.is_none());
        assert_eq!(g.fifo.enabled, 0);
        drop(g);
        assert!(r.device.stamp_memory.lock().is_none());
    }

    /// The teardown's stores reach the staged view together, before the mapping's release
    /// calls the host: `unmapMemory` sees the FIFO disabled and gone.
    #[test]
    fn teardown_fifos_publishes_before_the_unmap() {
        use std::sync::OnceLock;
        use std::sync::Weak;
        type Slot = Arc<OnceLock<Weak<DeviceCore<crate::testkit::FakePlatform>>>>;
        let slot: Slot = Arc::new(OnceLock::new());
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (s2, seen2) = (slot.clone(), seen.clone());
        let r = rig_with(move |b| {
            let inner = b.unmap_memory.clone().unwrap();
            b.unmap_memory = Some(Arc::new(move |t, o, l| {
                if let Some(d) = s2.get().and_then(Weak::upgrade) {
                    let a = d.root_fifo.atomics();
                    seen2.lock().push((a.enabled(), a.root_fifo().is_none()));
                }
                inner(t, o, l)
            }));
        });
        r.with_root_fifo(1, 2);
        slot.set(Arc::downgrade(&r.device)).ok().unwrap();
        r.device.teardown_fifos();
        assert_eq!(*seen.lock(), [(0, true)]);
    }

    #[test]
    fn enable_fifo_twice_dies() {
        let r = rig();
        r.with_root_fifo(1, 2);
        let d = r.device.clone();
        expect_death(DeviceDeath::FifoAlreadyEnabled.site(), move || {
            d.enable_fifo()
        });
    }

    // 0x1008 before the root FIFO exists is dropped.
    #[test]
    fn no_root_fifo_doorbell_dropped_and_fault_offset_zero() {
        let r = rig();
        r.device.set_fifo_written(0x40);
        assert_eq!(r.device.fifo_fault_offset(), 0);
    }

    #[test]
    fn teardown_fifos_child_in_slot_0_dies() {
        let r = rig();
        r.with_root_fifo(1, 2);
        let root = r.device.root_fifo.lock().root_fifo.clone();
        r.device.child_fifos.lock()[0] = root;
        let d = r.device.clone();
        assert_eq!(
            catch_death(|| d.teardown_fifos()).err(),
            Some(DeviceDeath::ChildInSlotZero.site())
        );
        r.device.child_fifos.lock()[0] = None;
    }

    // The root FIFO's teardown does not delete children whose id is at or past the current
    // maxFIFOCount (possible after lowering the version below 61 with children of 32 and up):
    // they keep running with no stamp memory, and a later stamp on one is a NULL write, a host
    // death.
    #[test]
    fn teardown_fifos_keeps_children_past_max() {
        let r = rig();
        r.set_ram_u32(APV_PAGE_SIZE + 0x400 + 2 * 0x14, 3);
        r.with_root_fifo(1, 2);
        assert!(r.device.create_fifo(3).unwrap());
        let mut f = r.device.features.load();
        f.max_fifo_count = 2;
        r.device.features.store(&f);
        r.device.teardown_fifos();
        assert!(r.device.child_fifos.lock()[3].is_some());
        assert!(r.device.stamp_memory.lock().is_none());
        f.max_fifo_count = 64;
        r.device.features.store(&f);
        assert!(r.device.delete_fifo(3).unwrap());
    }

    #[test]
    fn resume_fifo_without_root_clears_bit0_and_raises() {
        let r = rig();
        r.device.interrupt_fault_mask.store(0b11, Ordering::SeqCst);
        r.device.resume_fifo();
        assert_eq!(r.device.interrupt_fault_mask.load(Ordering::SeqCst), 0b10);
        assert_eq!(r.host.interrupts(), [0]);
        // Nothing left: no interrupt.
        r.device.interrupt_fault_mask.store(1, Ordering::SeqCst);
        r.device.resume_fifo();
        assert_eq!(r.device.interrupt_fault_mask.load(Ordering::SeqCst), 0);
        assert_eq!(r.host.interrupts(), [0]);
    }

    #[test]
    fn resume_fifo_healthy_root_keeps_bit0() {
        let r = rig();
        r.with_root_fifo(1, 2);
        r.device.interrupt_fault_mask.store(1, Ordering::SeqCst);
        r.device.resume_fifo();
        assert_eq!(r.device.interrupt_fault_mask.load(Ordering::SeqCst), 1);
        r.device.teardown_fifos();
    }

    #[test]
    fn contiguous_mapping_without_root_task_is_none() {
        let r = rig();
        let m = r.mark();
        assert!(r.device.new_contiguous_mapping(3, 0x4000).is_none());
        assert!(r.host_sels(m).is_empty());
    }

    #[test]
    fn contiguous_mapping_allocates_and_unmaps() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let m = r.mark();
        let map = r.device.new_contiguous_mapping(3, 0x8000).expect("mapping");
        let base = root_task(&r).expect("info").base_address;
        // After the root node's page at offset 0.
        assert_eq!(map.virtual_address(), base.wrapping_add(0x4000));
        assert_eq!(map.length(), 0x8000);
        let calls = r.host_calls(m);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, "mapMemory");
        assert_eq!(calls[0].args[2].as_u64(), Some(0x4000));
        drop(map);
        let calls = r.host_calls(m);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].sel, "unmapMemory");
        assert_eq!(calls[1].args[1].as_u64(), Some(0x4000));
        assert_eq!(calls[1].args[2].as_u64(), Some(0x8000));
        // The range is free again.
        let again = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
        assert_eq!(again.virtual_address(), base.wrapping_add(0x4000));
    }

    /// The root task's address space is never reset: a range a mapping leaked when it outlived
    /// its root task (`mapping_outliving_root_task_leaks`) stays taken under the next root task.
    #[test]
    fn root_va_is_never_reset() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let held = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
        r.device.set_root_page(Hold::NONE, 0);
        // No root task: the release skips the free, and offset 0x4000 stays taken.
        drop(held);
        r.device.set_root_page(Hold::NONE, 1);
        let base = root_task(&r).expect("info").base_address;
        let node = r.device.root_node.lock().root_node.expect("root node");
        assert_eq!(node.addr(), base);
        let next = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
        assert_eq!(next.virtual_address(), base.wrapping_add(0x8000));
    }

    #[test]
    fn failed_host_map_frees_the_range() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        r.platform.script().push("mapMemory", Step::value(false));
        assert!(r.device.new_contiguous_mapping(3, 0x4000).is_none());
        let base = root_task(&r).expect("info").base_address;
        let map = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
        assert_eq!(map.virtual_address(), base.wrapping_add(0x4000));
    }

    #[test]
    fn allocator_exhaustion_is_none() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let m = r.mark();
        assert!(r.device.new_contiguous_mapping(3, ROOT_TASK_SIZE).is_none());
        assert!(r.host_sels(m).is_empty());
    }

    // Mappings that outlive their root task (e.g. a guest display leaked by "Trying to set display
    // state page without root node") skip both the host unmap and the allocator's free: the
    // root-task range leaks for the device's life.
    #[test]
    fn mapping_outliving_root_task_leaks() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let map = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
        r.device.set_root_page(Hold::NONE, 0);
        let m = r.mark();
        drop(map);
        assert!(r.host_sels(m).is_empty());
    }

    #[test]
    fn range_mapping_passes_every_range() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let ranges = [
            PhysRange {
                physical_address: 0x8000,
                physical_length: 0x4000,
            },
            PhysRange {
                physical_address: 0x10000,
                physical_length: 0x4000,
            },
        ];
        let m = r.mark();
        let map = r
            .device
            .new_range_mapping(&ranges, 0x8000, true)
            .expect("mapping");
        let calls = r.host_calls(m);
        assert_eq!(calls[0].sel, "mapMemory");
        assert_eq!(calls[0].args[1], Arg::U32(2));
        // readOnly is not passed on.
        assert!(matches!(calls[0].args[3], Arg::Bool(false)));
        r.set_ram_u32(0x10000, 0x1234);
        r.platform.fake_vm().sync();
        assert_eq!(
            GuestPtr::new(map.virtual_address()).read_u32(0x4000, DeviceDeath::PhysicalReadCopy),
            0x1234
        );
    }

    #[test]
    fn range_mapping_unaligned_total_dies() {
        let r = rig();
        let d = r.device.clone();
        expect_death(DeviceDeath::UnalignedRangeMapping.site(), move || {
            let _ = d.new_range_mapping(&[], APV_PAGE_SIZE / 4, false);
        });
    }

    #[test]
    fn mappings_die_leaving_suspend_early() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let map = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
        r.device.run_state.store(RunState::LeavingSuspendEarly);
        let d = r.device.clone();
        let leaving = Some(DeviceDeath::MappingWhileLeavingSuspend.site());
        assert_eq!(
            catch_death(|| d.new_contiguous_mapping(3, 0x4000)).err(),
            leaving
        );
        assert_eq!(
            catch_death(|| d.new_range_mapping(&[], 0x4000, false)).err(),
            leaving
        );
        let va = map.virtual_address();
        assert_eq!(catch_death(|| d.unmap_range(va, 0x4000)).err(), leaving);
        r.device.run_state.store(RunState::Running);
        drop(map);
    }

    #[test]
    fn unmap_outside_root_task_dies() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let base = root_task(&r).expect("info").base_address;
        let d = r.device.clone();
        assert_eq!(
            catch_death(|| d.unmap_range(base.wrapping_add(ROOT_TASK_SIZE), 0x4000)).err(),
            Some(DeviceDeath::UnmapOutsideRootTask.site())
        );
    }

    /// The unmap checks the offset before it takes the root task's lock, so an out-of-range
    /// unmap under the held lock (a host block that runs under it) dies rather than waiting on
    /// the lock.
    #[test]
    fn unmap_dies_before_taking_the_root_task_lock() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        let base = root_task(&r).expect("info").base_address;
        let d = r.device.clone();
        let _held = r.device.root_task.lock();
        assert_eq!(
            catch_death(|| d.unmap_range(base.wrapping_add(ROOT_TASK_SIZE), 0x4000)).err(),
            Some(DeviceDeath::UnmapOutsideRootTask.site())
        );
    }

    #[test]
    fn nil_map_block_dies() {
        let r = rig_with(|b| b.map_memory = None);
        let d = r.device.clone();
        expect_death(DeviceDeath::NilBlock("mapMemory").site(), move || {
            d.set_root_page(Hold::NONE, 1)
        });
    }

    #[test]
    fn read_past_end_of_page_raises() {
        let r = rig();
        let mut dst = vec![0u8; 0x4001];
        let raised = r
            .device
            .read_from_physical_address(OnFault::Return, 0, &mut dst)
            .expect_err("raises");
        assert_eq!(raised.id(), RaiseId::Device(DeviceRaise::ReadPastPage));
        let mut dst = [0u8; 8];
        assert!(
            r.device
                .read_from_physical_address(OnFault::Return, 0x3ffc, &mut dst)
                .is_err()
        );
        // With no handler it dies at the raise.
        let mut dst = [0u8; 8];
        assert!(
            catch_death(|| r
                .device
                .read_from_physical_address(OnFault::Die, 0x3ffc, &mut dst))
            .is_err()
        );
    }

    #[test]
    fn read_through_read_block() {
        let r = rig();
        r.set_ram_u32(0x8010, 0xdead_beef);
        let mut dst = [0u8; 4];
        let m = r.mark();
        assert!(
            r.device
                .read_from_physical_address(OnFault::Die, 0x8010, &mut dst)
                .unwrap()
        );
        assert_eq!(u32::from_le_bytes(dst), 0xdead_beef);
        assert_eq!(r.host_sels(m), ["readMemory"]);
    }

    #[test]
    fn read_through_contiguous_mapping() {
        let r = rig_with(|b| b.read_memory = None);
        r.set_ram_u32(0x8010, 0xfeed_f00d);
        let mut dst = [0u8; 4];
        // No root task: the mapping fails.
        assert!(
            !r.device
                .read_from_physical_address(OnFault::Die, 0x8010, &mut dst)
                .unwrap()
        );
        r.device.set_root_page(Hold::NONE, 1);
        let m = r.mark();
        assert!(
            r.device
                .read_from_physical_address(OnFault::Die, 0x8010, &mut dst)
                .unwrap()
        );
        assert_eq!(u32::from_le_bytes(dst), 0xfeed_f00d);
        assert_eq!(r.host_sels(m), ["mapMemory", "unmapMemory"]);
    }

    /// The host's block raising an `NSException` (a scripted `Step::Raise`).
    fn host_raise() -> Step {
        Step::raise("HostBlockRaise", "the host refused")
    }

    /// A host block that raises ends the process at the call, under a handler or not.
    mod host_block_raises {
        use super::*;
        use crate::raise::death::DeathKind;
        use crate::raise::death::DeathSite;

        /// The call a raising block dies at.
        fn host_raised<T>(d: Result<T, DeathSite>) -> &'static str {
            let d = d.err().expect("a death");
            assert_eq!(d.kind, DeathKind::HostRaised);
            d.file
        }

        #[test]
        fn a_raising_map_dies_at_the_call() {
            let r = rig();
            r.device.set_root_page(Hold::NONE, 1);
            r.platform.script().push("mapMemory", host_raise());
            let d = catch_death(|| {
                let _ = r.device.new_contiguous_mapping(3, 0x4000);
            });
            assert_eq!(host_raised(d), "mapMemory");
        }

        #[test]
        fn a_raising_unmap_dies_at_the_release() {
            let r = rig();
            r.device.set_root_page(Hold::NONE, 1);
            let map = r.device.new_contiguous_mapping(3, 0x4000).expect("mapping");
            r.platform.script().push("unmapMemory", host_raise());
            let d = catch_death(move || drop(map));
            assert_eq!(host_raised(d), "unmapMemory");
        }

        #[test]
        fn a_raising_read_block_dies_at_the_call() {
            let r = rig();
            r.platform.script().push("readMemory", host_raise());
            let d = catch_death(|| {
                let mut dst = [0u8; 4];
                let _ = r
                    .device
                    .read_from_physical_address(OnFault::Return, 0x8010, &mut dst);
            });
            assert_eq!(host_raised(d), "readMemory");
        }

        #[test]
        fn a_raising_create_task_in_set_root_page_dies() {
            let r = rig();
            r.platform.script().push("createTask", host_raise());
            let d = r.device.clone();
            assert_eq!(
                host_raised(catch_death(move || d.set_root_page(Hold::NONE, 1))),
                "createTask"
            );
        }
    }

    /// A refused root FIFO logs the FIFO's reason, then the device's refusal.
    #[test]
    fn refused_root_fifo_logs_start_offset_then_failure() {
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        r.device.set_fifo_base_page(2);
        r.device.set_fifo_length(0x4000);
        r.device.set_fifo_start(0x4000);
        let lines = Lines::capture();
        r.device.enable_fifo();
        assert_eq!(r.device.root_fifo.lock().fifo.enabled, 0);
        assert_eq!(
            lines.all(),
            ["root FIFO not created: the FIFO refused its registers"]
        );
    }
}
