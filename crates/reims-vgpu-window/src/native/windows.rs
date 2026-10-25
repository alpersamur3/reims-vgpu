//! The window on Windows: none yet. A display made with a window config gets no window here and
//! draws nothing; a host on Windows reads a display's frames through a frame target
//! (`rvg_frame_target_*`), as QEMU's console does. What the shared window code asks of this
//! side keeps Linux's shapes, so the Vulkan platform's sink (`reims-vgpu`'s
//! `window/vulkan/linux.rs`) serves a later Win32 window unchanged: a presenter over the window's
//! Vulkan surface (`VK_KHR_win32_surface`).

use std::sync::Arc;

use crate::handle::Window;
use crate::present::{Presenter, VulkanTarget};
use crate::screens::HostScreen;

/// What draws a display's current frame into a window: `reims-vgpu`'s, over the display, which it
/// holds weakly. Linux's trait, unchanged.
pub trait FrameSink: Send + Sync {
    /// A presenter over `target`'s window: one that makes the window's surface on the sink's
    /// Vulkan instance and a swapchain over it. `None` where there is none (the display has
    /// ended, or this host has no swapchain): the window then stays empty.
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>>;
}

/// What a window's render thread would draw with: nothing on Windows, where no window opens.
#[derive(Debug, Default)]
pub(crate) struct Surface;

impl Surface {
    /// The window's surface.
    pub(crate) fn new() -> Surface {
        Surface
    }

    /// A new frame or layout to draw: nothing to draw into.
    pub(crate) fn request_draw(&self) {}

    /// The window's end: nothing to end.
    pub(crate) fn close(&self) {}
}

/// The window's opening: refused, logged once per window. The display stays usable through a
/// frame target.
pub(crate) fn open(window: &Arc<Window>) {
    log::warn!(
        target: "reims_vgpu_window",
        "no window opens on Windows (window {}): read the display through a frame target",
        window.id()
    );
}

/// The guest of window `id` set a mode: no window to resize.
pub(crate) fn guest_mode_changed(_id: u64) {}

/// A cursor refresh of window `id`: no window to draw it in.
pub(crate) fn cursor_changed(_id: u64) {}

/// The close of window `id`: no window to close.
pub(crate) fn close(_id: u64) {}

/// The host's screens: none listed, as no window opens on them.
pub(crate) fn host_screens() -> Vec<HostScreen> {
    Vec::new()
}
