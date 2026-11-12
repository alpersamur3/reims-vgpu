//! The OS services: core's threads, clocks, host VM and IOSurface, each OS's own behind one
//! facade (`OsClock`, `OsThreads`, `OsVm`, `OsSurface`, `OsSurfaces`), and what this platform
//! reads of a surface beyond core's trait, written against it (`surface.rs`).
//!
//! - **macOS** (`os/macos.rs`): the metal crate's services, as they are, and IOSurface itself.
//! - **Windows** (`os/windows.rs`): the clocks (interrupt time, precise system time), threads,
//!   a host VM over Win32 virtual memory whose ranges lie at or above 4 GiB and that makes views
//!   of the guest's pages by mapping their sections over placeholders (the backing table), and
//!   IOSurface's stand-in over that memory (ASoftware's port).
//! - **Linux** (`os/linux.rs`): the clocks (`CLOCK_BOOTTIME`, `CLOCK_REALTIME`), threads, a host
//!   VM over `mmap` that makes views of the guest's pages by mapping their files (the backing
//!   table), and IOSurface's stand-in over that memory.
//!
//! Either way the platform's VM reports the pages it changes to the runtime
//! ([`crate::remap::VulkanVm`]), and a display's window presents through a swapchain over the
//! window system's Vulkan extensions (`wsi.rs`), each OS naming its own surface extensions and
//! its loader, and fetching an image's drawable ahead where the driver needs it
//! (`Drawables`: MoltenVK's, on macOS).

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
mod surface;
#[cfg(windows)]
mod windows;
mod wsi;

#[cfg(any(target_os = "linux", target_os = "android"))]
use linux as host_os;
#[cfg(target_os = "macos")]
use macos as host_os;
#[cfg(windows)]
use windows as host_os;

use std::sync::Arc;

/// The OS's handle to a file whose bytes back a range of the guest's RAM: a file descriptor on
/// unix, a handle (a section's, `CreateFileMappingW`) on Windows.
#[cfg(unix)]
pub type HostFile = std::os::fd::OwnedFd;
/// The OS's handle to a file whose bytes back a range of the guest's RAM: a file descriptor on
/// unix, a handle (a section's, `CreateFileMappingW`) on Windows.
#[cfg(windows)]
pub type HostFile = std::os::windows::io::OwnedHandle;

pub(crate) use host_os::{
    Drawables, METAL_LEVEL, TWINS_READ_ONLY, allocate_aliasable, allocate_zeros, dma_buf_refusal,
    host_memory, import_view, plane_shape, shown_at, surface_memory_hold,
};
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) use linux::allocate_deferred;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) use linux::{
    ViewCounts, give_back_view, idle_views_enabled, view_counts, view_makers, view_mappings,
    views_census_on,
};

/// No deferred views on this OS: the caller's own range (`os/linux/deferred.rs`).
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) fn allocate_deferred(
    _spans: &[(u64, u64)],
    _length: u64,
) -> Option<reims_vgpu_core::platform::os::vm::HostAddr> {
    None
}
pub use host_os::{
    GuestMemoryLease, OsClock, OsSurface, OsSurfaces, OsThreads, OsVm, default_gpu_core_count,
    default_registry_id, install_process_services, lease_guest_memory,
};
#[cfg(test)]
pub(crate) use host_os::{PAGE, mapped, writable};
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
pub(crate) use linux::IDLE_TEST_ON;
#[cfg(all(test, target_os = "macos"))]
pub(crate) use macos::MATRIX_KEY;
pub use wsi::{add_window_extensions, present_fences};

pub(crate) use surface::{
    MadeOver, READ_ONLY_DIRECTION, base_in_pages, dropped_write, gpu_read_only, in_runs,
    memory_kind, over_guest_pages, page_runs, plane, plane_facts, surface_alias,
    surface_bytes_memory, surface_memory, surface_offset, write_surface_bytes, ycbcr_facts,
};

/// One range of the guest's RAM as the host maps it, and the file behind it: `length` bytes at
/// `host`, a shared mapping (`MAP_SHARED`) of `fd` from byte `offset`. A device's lease of the
/// guest's RAM ([`lease_guest_memory`]) is made of these.
#[derive(Clone, Debug)]
pub struct GuestRamFile {
    /// The host address of the range's first byte.
    pub host: u64,
    /// The range's length in bytes.
    pub length: u64,
    /// The file.
    pub fd: Arc<HostFile>,
    /// The file offset of the range's first byte.
    pub offset: u64,
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_core::platform::os::vm::HostVm as _;

    // The surface tests' helpers, which the crate's tests name here.
    pub(crate) use super::surface::tests::*;
    use super::*;

    /// The tests' page is the host's
    #[test]
    fn the_tests_page_is_the_hosts() {
        assert_eq!(OsVm::new().page_size(), PAGE);
    }
}
