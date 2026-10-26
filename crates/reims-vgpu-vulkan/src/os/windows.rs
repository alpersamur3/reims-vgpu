//! The OS services on Windows (ASoftware's port):
//! the clocks ([`WindowsClock`]), threads ([`WindowsThreads`]), Mach VM's calls over Win32
//! virtual memory ([`WindowsVm`]), the IOSurface stand-in ([`WindowsSurfaces`]) and what the
//! window system's Vulkan extensions take here (`crate::os`'s `wsi`: `VK_KHR_win32_surface` and
//! `vulkan-1.dll`). No Objective-C runs here, as on Linux.
//!
//! **What this arm does** (section-backed views, IOSurface stand-ins):
//! - Real: the clocks, the threads, fresh zero-filled ranges at or above 4 GiB (`VirtualAlloc2`
//!   with an address requirement, as core's `GuestPtr` dies on any address below
//!   `ZERO_REGION_END`), protection, deallocation and the readable probe.
//! - Real: views of the guest's pages. The guest's RAM is a view of a pagefile section the host
//!   names in its memory map (`CreateFileMappingW`); `remap_shared_overwrite` maps the pages'
//!   section over placeholders carved out of the target (`MapViewOfFile3` with
//!   `MEM_REPLACE_PLACEHOLDER`), looked up in the process's backing table ([`files`]). The VM's
//!   own aliasable ranges come from a library-owned section the same way.
//! - Real: IOSurface stand-ins, as on Linux ([`surface`]): a surface of its own memory is a view
//!   of a library-owned section, registered so that no `deallocate` frees it; one over several
//!   ranges of the guest's pages is a view of their sections.
//! - Imports are of the bytes themselves where they are read and write: the runtime's Windows
//!   importer takes a host pointer with `VK_EXT_external_memory_host`'s `HOST_ALLOCATION` handle
//!   type, which the RTX 4080's Windows driver takes for committed memory and for a section's
//!   view alike (measured). Read-only memory it refuses, so that is imported through a
//!   writable twin of its sections, as on Linux ([`import`]).

mod clock;
mod files;
mod import;
mod surface;
mod threads;
mod vm;

use std::ffi::CStr;
use std::ptr::NonNull;

use ash::vk;
use metal2vulkan_runtime as rt;

pub use clock::WindowsClock;
pub use surface::{WindowsSurface, WindowsSurfaces};
pub use threads::WindowsThreads;
pub use vm::WindowsVm;

pub(crate) use import::{TWINS_READ_ONLY, allocate_zeros, import_view, shown_at};
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

/// What a device holds of the guest's RAM while it lives: the registrations, in the backing
/// table, of the sections the host names behind the memory map's ranges, which views of the
/// guest's pages are made of ([`files::lease_guest_memory`]).
pub use files::{GuestMemoryLease, lease_guest_memory};

/// A fresh range of `len` bytes that a view can be made of: a view of a pagefile section of its
/// own over a placeholder, recorded in the backing table ([`vm::allocate_aliasable`]).
pub(crate) use vm::allocate_aliasable;

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

