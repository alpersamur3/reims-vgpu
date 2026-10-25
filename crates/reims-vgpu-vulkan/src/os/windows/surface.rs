//! IOSurface's stand-in on Windows: none made yet. Linux's stand-in (`os/linux/surface.rs`) is
//! memory a view can be made of (a file of its own, or a view of the guest's pages); on Windows
//! that memory is a section and its views (`CreateFileMappingW`, `MapViewOfFile3`), which the
//! VM does not make yet (`vm.rs`). Until then `create` and `create_backing` answer none, which
//! core answers to the guest as IOSurface's refusal, and [`WindowsSurface`] has no value: the
//! helpers over a surface below are never reached.

use std::sync::Once;

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::os::surface::{
    BackingSurfaceSpec, IoSurfaces, Surface, SurfaceSpec,
};
use reims_vgpu_core::platform::os::vm::KernReturn;
use reims_vgpu_core::platform::{Handle, ObjId};

use crate::os::MadeOver;
use crate::platform::VulkanPlatform;

/// An IOSurface stand-in: none can be made yet (module docs).
#[derive(Debug)]
pub enum WindowsSurface {}

impl Handle for WindowsSurface {
    fn clone_ref(&self) -> Self {
        match *self {}
    }

    fn id(&self) -> ObjId {
        match *self {}
    }
}

impl Surface for WindowsSurface {
    fn retain_count(&self) -> i64 {
        match *self {}
    }

    fn pixel_format(&self) -> u32 {
        match *self {}
    }

    fn width(&self) -> u64 {
        match *self {}
    }

    fn height(&self) -> u64 {
        match *self {}
    }

    fn bytes_per_element(&self) -> u64 {
        match *self {}
    }

    fn bytes_per_row(&self) -> u64 {
        match *self {}
    }

    fn plane_count(&self) -> u64 {
        match *self {}
    }

    fn allows_pixel_size_casting(&self) -> bool {
        match *self {}
    }

    fn protection_options(&self) -> u64 {
        match *self {}
    }

    fn lock(&self, _options: u32) -> KernReturn {
        match *self {}
    }

    fn unlock(&self, _options: u32) -> KernReturn {
        match *self {}
    }

    fn base_address(&self) -> HostPtr {
        match *self {}
    }

    fn alloc_size(&self) -> u64 {
        match *self {}
    }

    fn lookup_by_port(&self) -> Option<Self> {
        match *self {}
    }
}

/// Plane `plane` of `surface`: never reached (module docs).
pub(crate) fn plane_at(surface: &WindowsSurface, _plane: u64) -> Option<(u64, u64)> {
    match *surface {}
}

/// The width, height and bytes per element of plane `plane` of `surface`: never reached.
pub(crate) fn plane_shape(surface: &WindowsSurface, _plane: u64) -> Option<(u64, u64, u64)> {
    match *surface {}
}

/// The YCbCr matrix `surface` names: never reached.
pub(crate) fn ycbcr_matrix(
    surface: &WindowsSurface,
) -> Option<metal2vulkan_runtime::SurfaceYCbCrMatrix> {
    match *surface {}
}

/// What `surface` was made over, `change` given it: never reached.
pub(crate) fn recorded_mut<T>(
    surface: &WindowsSurface,
    _change: impl FnOnce(&mut MadeOver) -> T,
) -> Option<T> {
    match *surface {}
}

/// A hold on the memory `surface` mapped: never reached.
pub(crate) fn surface_memory_hold(surface: &WindowsSurface) -> impl Send + 'static {
    let () = match *surface {};
}

/// IOSurface creation: refused for now (module docs), logged once.
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsSurfaces;

impl WindowsSurfaces {
    /// The process's IoSurfaces.
    pub fn new() -> WindowsSurfaces {
        WindowsSurfaces
    }
}

/// Logs, once, that no stand-in is made.
fn refused() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "no IOSurface stand-ins on Windows yet: every surface is refused"
        );
    });
}

impl IoSurfaces<VulkanPlatform> for WindowsSurfaces {
    fn create(&self, _spec: &SurfaceSpec) -> Option<WindowsSurface> {
        refused();
        None
    }

    fn create_backing(&self, _spec: &BackingSurfaceSpec) -> Option<WindowsSurface> {
        refused();
        None
    }

    fn describe(&self, spec: &SurfaceSpec) -> String {
        format!("{spec:?}")
    }
}
