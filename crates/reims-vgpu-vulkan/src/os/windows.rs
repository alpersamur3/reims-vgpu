//! The OS services on Windows (ASoftware's port):
//! the clocks ([`WindowsClock`]), threads ([`WindowsThreads`]), Mach VM's calls over Win32
//! virtual memory ([`WindowsVm`]), the IOSurface stand-in ([`WindowsSurfaces`]) and what the
//! window system's Vulkan extensions take here (`crate::os`'s `wsi`: `VK_KHR_win32_surface` and
//! `vulkan-1.dll`). No Objective-C runs here, as on Linux.
//!
//! **What this arm does and refuses** (M3a.3, section-backed views):
//! - Real: the clocks, the threads, fresh zero-filled ranges at or above 4 GiB (`VirtualAlloc2`
//!   with an address requirement, as core's `GuestPtr` dies on any address below
//!   `ZERO_REGION_END`), protection, deallocation and the readable probe.
//! - Real: views of the guest's pages. The guest's RAM is a view of a pagefile section the host
//!   names in its memory map (`CreateFileMappingW`); `remap_shared_overwrite` maps the pages'
//!   section over placeholders carved out of the target (`MapViewOfFile3` with
//!   `MEM_REPLACE_PLACEHOLDER`), looked up in the process's backing table ([`files`]). The VM's
//!   own aliasable ranges come from a library-owned section the same way.
//! - Refused: IOSurface stand-ins (`create` answers none), a later step.
//! - Imports are of the bytes themselves, as on macOS: the runtime's Windows importer takes a
//!   host pointer with `VK_EXT_external_memory_host`'s `HOST_ALLOCATION` handle type, which the
//!   RTX 4080's Windows driver takes for committed memory and for a section's view alike
//!   (measured, M3a.0).

mod clock;
mod files;
mod surface;
mod threads;
mod vm;

use std::ffi::CStr;
use std::ptr::NonNull;

use ash::vk;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _, KernReturn};

pub use clock::WindowsClock;
pub use surface::{WindowsSurface, WindowsSurfaces};
pub use threads::WindowsThreads;
pub use vm::WindowsVm;

pub(crate) use surface::{plane_at, plane_shape, recorded_mut, surface_memory_hold, ycbcr_matrix};
#[cfg(test)]
pub(crate) use vm::tests::{PAGE, mapped, writable};

/// The instance's surface extension a display's window may present over, besides
/// `VK_KHR_surface`, asked for where the loader lists it (`crate::os`'s `wsi`): Win32's.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::khr::win32_surface::NAME];

/// The device extensions a window's swapchain needs besides `VK_KHR_swapchain`: none.
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 0] = [];

/// The Vulkan loader, opened by its usual name (`vulkan-1.dll`), as the runtime opens it.
pub(crate) fn open_loader() -> Result<ash::Entry, ash::LoadingError> {
    // SAFETY: opening the loader runs its initialization, and it is kept open while the entry
    // lives.
    unsafe { ash::Entry::load() }
}

/// What fetches a swapchain image's drawable ahead of its frame: nothing here, as on Linux.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Drawables {}

impl Drawables {
    /// None on Windows.
    pub(crate) fn load(_raw: &rt::Device) -> Option<Drawables> {
        None
    }

    /// Never called: there is no value.
    ///
    /// # Safety
    ///
    /// None needed.
    pub(crate) unsafe fn fetch(&self, _image: vk::Image) -> bool {
        match *self {}
    }
}

/// The clocks.
pub type OsClock = WindowsClock;
/// The threads.
pub type OsThreads = WindowsThreads;
/// Mach VM's calls.
pub type OsVm = WindowsVm;
/// An IOSurface: the stand-in's.
pub type OsSurface = WindowsSurface;
/// IOSurface creation: the stand-in's.
pub type OsSurfaces = WindowsSurfaces;

/// Installs what the process's platform needs before it runs: nothing on Windows, which has no
/// open-file limit to raise for the stand-ins' memory (a section handle is no file descriptor).
pub fn install_process_services() {}

/// The macOS release whose Metal the runtime's devices answer as where releases differ: macOS
/// 26.7, as on Linux, so a Windows host answers the guest as the suite's Mac does.
pub const METAL_LEVEL: Option<rt::MetalLevel> = Some(rt::MetalLevel::MacOS26_7);

/// The registry ID the platform answers for the runtime's default device in place of its own:
/// none, as there is no Metal here to ask.
pub fn default_registry_id() -> Option<u64> {
    None
}

/// The GPU core count Metal answers for the default device: none, as there is no Metal here.
pub fn default_gpu_core_count() -> Option<u64> {
    None
}

/// What a device holds of the guest's RAM while it lives: nothing yet. Section-backed guest RAM
/// (the host's `CreateFileMappingW` section behind each range) is what views will be made of;
/// until the memory map's ABI carries a section handle, a device holds none.
#[derive(Debug)]
pub struct GuestMemoryLease;

/// The lease of the guest's RAM a device holds: nothing on Windows yet ([`GuestMemoryLease`]).
pub fn lease_guest_memory(_: impl IntoIterator<Item = super::GuestRamFile>) -> GuestMemoryLease {
    GuestMemoryLease
}

/// A fresh range of `len` bytes that a view can be made of: on Windows, for now, any fresh
/// committed range at or above 4 GiB, as on Mach; a view of it waits for section-backed ranges.
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    WindowsVm::new().allocate_anywhere(len)
}

/// A fresh range of `len` bytes of zeros that a view can be made of: [`allocate_aliasable`]'s.
pub(crate) fn allocate_zeros(len: u64) -> Result<HostAddr, KernReturn> {
    allocate_aliasable(len)
}

/// Whether the runtime imports read-only memory through a writable twin of its files: not on
/// Windows, where it imports the bytes themselves ([`import_view`]).
pub(crate) const TWINS_READ_ONLY: bool = false;

/// The memory the runtime imports for some bytes: on Windows those bytes themselves, which the
/// runtime imports by their host pointer (`HOST_ALLOCATION`).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the GPU's writes through the import are kept where a read-only import drops
    /// them: never here, as the bytes themselves are imported.
    pub(crate) fn writes_kept(&self) -> bool {
        false
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves: `inner`, as it is, and nothing, as the runtime imports the bytes.
    pub(crate) fn release(
        self,
        _original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        (inner, None)
    }
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes ([`ImportView`]).
pub(crate) fn import_view(at: NonNull<u8>, _length: u64) -> ImportView {
    ImportView { at }
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]): the bytes
/// alone.
///
/// # Safety
///
/// `rt::HostMemory::new`'s contract, for the memory at `at`.
pub(crate) unsafe fn host_memory(at: NonNull<u8>) -> rt::HostMemory {
    // SAFETY: the caller's.
    unsafe { rt::HostMemory::new(at) }
}

/// Why this thread's last dma-buf source made none: never, on Windows, which has no dma-bufs.
pub(crate) fn dma_buf_refusal() -> Option<String> {
    None
}

/// The address whose bytes the runtime's memory at `at` shows: `at`, as there are no twins.
pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    at
}
