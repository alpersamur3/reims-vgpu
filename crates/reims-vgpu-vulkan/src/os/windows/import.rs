//! What the runtime imports for host memory on Windows: the memory itself where every page of it
//! is mapped read and write, and otherwise a writable twin of it.
//!
//! NVIDIA's Windows driver refuses a host pointer to read-only pages
//! (`VK_EXT_external_memory_host`'s `HOST_ALLOCATION`), and the runtime reports it as out of
//! device memory (measured on the RTX 4080, driver 0x98928000, 2026-10-26: a committed range
//! imports, the same range again right after its buffer went imports, and the same range after
//! `VirtualProtect(PAGE_READONLY)` is refused). Read-only memory is what Metal reads where Linux
//! drivers refuse it too: aliases of a read-only replace or heap, and surfaces the GPU only reads.
//! So, as on Linux, the runtime is given a **twin** for those: a second view of the same sections,
//! read and write, over a placeholder of its own ([`import_view`]). The GPU then reads what Metal's
//! reads, but its writes reach those pages, where Metal drops them ([`ImportView::writes_kept`]):
//! every command that may write through an object over such an import dies where it is encoded,
//! naming it (`crate::gpu::writes::kept_write`).
//!
//! The block of zeros a read-only alias maps past its ranges' pages is a section of zeros all
//! such aliases share ([`allocate_zeros`]). Its twin is private zeros instead, so the GPU's writes
//! there stay in that twin and never reach the block.
//!
//! A buffer made over a twin answers the address it was made over as its contents
//! ([`shown_at`]). The twin goes once the runtime frees the import (its keep-alive,
//! [`ImportView::release`]).

use std::collections::BTreeMap;
use std::os::windows::io::OwnedHandle;
use std::ptr::NonNull;
use std::sync::Arc;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn};
use reims_vgpu_core::sync::Mutex;
use windows_sys::Win32::System::Memory::PAGE_READWRITE;

use super::files;
use super::vm;

/// Whether the runtime imports read-only memory through a writable twin of its pages: on
/// Windows, as NVIDIA's driver refuses read-only pages (module docs).
pub(crate) const TWINS_READ_ONLY: bool = true;

/// Every live twin, by its first byte: its length and the first byte of the memory it twins.
static TWINS: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// The sections of zeros ([`allocate_zeros`]), held for the process.
static ZERO_SECTIONS: Mutex<Vec<Arc<OwnedHandle>>> = Mutex::new(Vec::new());

/// A fresh range of `len` bytes of zeros that a view can be made of, as
/// [`vm::allocate_aliasable`] makes, whose section is kept for the process: an import's twin of
/// it is private zeros, never the section (module docs).
pub(crate) fn allocate_zeros(len: u64) -> Result<HostAddr, KernReturn> {
    let at = vm::allocate_aliasable(len)?;
    let first = at.addr().to_u64();
    if let Some(piece) = files::pieces(first, vm::page_size()).and_then(|p| p.into_iter().next()) {
        ZERO_SECTIONS.lock().push(piece.backing.section);
    }
    Ok(at)
}

/// Whether `backing` is of a section of zeros.
fn zeros(backing: &files::Backing) -> bool {
    ZERO_SECTIONS
        .lock()
        .iter()
        .any(|z| Arc::ptr_eq(z, &backing.section))
}

/// The memory the runtime imports for `length` bytes at `at` (module docs).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
    twin: Option<Twin>,
}

/// A writable twin's address range, unmapped when it drops.
#[derive(Debug)]
struct Twin {
    first: u64,
    length: u64,
}

impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.lock().remove(&self.first);
        // The twin's own range, which the runtime let go of (it freed the import and dropped its
        // keep-alive, or the call that would import it was refused).
        vm::drop_twin(self.first, self.length);
    }
}

/// An address moved into a deallocator, which may run on any thread.
struct SendPtr(NonNull<u8>);

// SAFETY: an address only, handed back to the deallocator it was given for.
unsafe impl Send for SendPtr {}

impl SendPtr {
    /// The address.
    fn get(self) -> NonNull<u8> {
        self.0
    }
}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the GPU's writes through the import are kept where Metal drops them: it is a twin
    /// of memory mapped read-only, the guest's pages or zeros (module docs).
    pub(crate) fn writes_kept(&self) -> bool {
        self.twin.is_some()
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves. With a twin: `inner`, given `original`, the address the bytes were
    /// given at, and the twin, which the runtime drops once it freed the import, so the twin is
    /// unmapped after it. Without one, `inner` as it is, and nothing.
    pub(crate) fn release(
        self,
        original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let Some(twin) = self.twin else {
            return (inner, None);
        };
        let original = SendPtr(original);
        let inner = inner.map(|inner| -> rt::Deallocator {
            Box::new(move |_, length| inner(original.get(), length))
        });
        (inner, Some(Box::new(twin)))
    }
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes where every page of
/// them is mapped read and write or one has no section behind it (which the runtime then refuses
/// as it refuses any import there), and otherwise a writable twin of their pages, at the same
/// offset in its first page (module docs).
pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let plain = ImportView { at, twin: None };
    let start = at.as_ptr().addr().to_u64();
    let page = vm::page_size();
    let first = start & !(page - 1);
    let Some(end) = start
        .checked_add(length.max(1))
        .and_then(|e| e.checked_next_multiple_of(page))
    else {
        return plain;
    };
    let Some(pieces) = files::pieces(first, end - first) else {
        return plain;
    };
    if pieces.iter().all(|p| p.prot == PAGE_READWRITE) {
        return plain;
    }
    let Some(base) = vm::twin(end - first, &pieces, zeros) else {
        log::debug!(
            target: "reims_vgpu_vulkan::os",
            "no twin of {:#x} bytes at {first:#x}: imported as they are",
            end - first
        );
        return plain;
    };
    let twin = Twin {
        first: base,
        length: end - first,
    };
    TWINS.lock().insert(base, (end - first, first));
    let Some(twin_at) = NonNull::new(vm::ptr(base + (start - first)).cast::<u8>()) else {
        return plain;
    };
    ImportView {
        at: twin_at,
        twin: Some(twin),
    }
}

/// The address whose bytes the runtime's memory at `at` shows: `at`, but for an address in a
/// writable twin, the twinned memory's at the same offset (module docs).
pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    let addr = at.as_ptr().addr().to_u64();
    let twins = TWINS.lock();
    let Some((&twin, &(length, first))) = twins.range(..=addr).next_back() else {
        return at;
    };
    if addr >= twin + length {
        return at;
    }
    NonNull::new(vm::ptr(first + (addr - twin)).cast::<u8>()).unwrap_or(at)
}
