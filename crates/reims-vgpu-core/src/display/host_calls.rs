//! What a display calls in the host: the host's handlers it keeps with its queue, and the mode
//! change it dispatches onto that queue. The handler and image types are the host's.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::display::descriptor::DisplayDescriptor;
use crate::display::{DeathKind, DisplayCoord, DisplayCore, DisplayLocked};
use crate::host::HostQueue;
use crate::num::Truncate;
use crate::platform::Platform;
use crate::rt::Hold;

/// The pixels of a cursor glyph: RGBA, packed rows (architecture 10).
#[derive(Clone)]
pub struct CursorGlyph {
    /// Pixels wide.
    pub width: u32,
    /// Pixels high.
    pub height: u32,
    /// `width * 4` bytes per row. Empty when [`CursorGlyph::image`] holds the pixels.
    pub rgba: Arc<[u8]>,
    /// The host's image the copy wrote the pixels into, when the descriptor has a
    /// [`GlyphImageMaker`]: the handler receives it +0, and the dispatched block's reference keeps
    /// it for the call. The public APIs give no maker: their hosts receive the pixels in `rgba`
    /// (the C API's `RvgCursorGlyph`).
    pub image: Option<GlyphImageRef>,
}

impl std::fmt::Debug for CursorGlyph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorGlyph")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("rgba", &self.rgba)
            .field("image", &self.image.as_ref().map(|i| i.host_ptr()))
            .finish()
    }
}

/// A host's image for a cursor glyph, made on the device thread under the guest display's mutex,
/// before the copy that writes its pixels. Only the tests make one: the public APIs give no
/// [`GlyphImageMaker`].
pub trait GlyphImage: HostObject {
    /// `write` runs over the image's `width * height` pixel words, or over `None` when the image
    /// has no buffer; its raise is the answer.
    #[expect(
        clippy::type_complexity,
        reason = "the writer takes the hold it runs under"
    )]
    fn write_bitmap_data(
        &self,
        hold: Hold<'_>,
        write: &mut dyn FnMut(Hold<'_>, Option<&mut [u32]>) -> crate::raise::RaiseResult<()>,
    ) -> crate::raise::RaiseResult<()>;
    /// The host's pointer to the image (+0), which a C host's handler receives.
    fn host_ptr(&self) -> crate::host::HostPtr;
}

/// A pointer to a [`GlyphImage`] (+0: valid while a reference to it lives).
pub type GlyphImageRef = Arc<dyn GlyphImage>;

/// Makes the host's image for a `(width, height)` glyph: the image and the reference its making
/// returned (+1, which the cursor update's holder keeps to its end), or `None` for nil.
pub type GlyphImageMaker =
    Arc<dyn Fn(u16, u16) -> Option<(GlyphImageRef, HostRetained)> + Send + Sync>;

/// The mode change handler: `(sizeInPixels, pixelFormat)`.
pub type ModeChangeHandler = Arc<dyn Fn(DisplayCoord, u32) + Send + Sync>;
/// The newFrame handler.
pub type NewFrameEventHandler = Arc<dyn Fn() + Send + Sync>;
/// The cursor glyph handler: `(glyph, hotSpot)`.
pub type CursorGlyphHandler = Arc<dyn Fn(&CursorGlyph, DisplayCoord) + Send + Sync>;
/// The cursor show handler. `show` is the guest's raw `BOOL` byte, unnormalised: the C API hands
/// the host that byte as it is (2 stays 2).
pub type CursorShowHandler = Arc<dyn Fn(u8) + Send + Sync>;
/// The cursor move handler.
pub type CursorMoveHandler = Arc<dyn Fn() + Send + Sync>;
/// Not an SDK handler: the guest display's sleep, `true`, and wake, `false` (FIFO opcode 9).
pub type SleepHandler = Arc<dyn Fn(bool) + Send + Sync>;

/// A host object a reference keeps alive, reached through a pointer that holds none of its own.
pub trait HostObject: Send + Sync {
    /// A reference the caller owns; dropping it releases the object.
    fn retain(&self) -> HostRetained;
}

/// One reference to a host object ([`HostObject::retain`]); its drop releases it.
pub type HostRetained = Box<dyn std::any::Any + Send>;

/// A pointer to a host object (+0).
pub type HostObjectRef = Arc<dyn HostObject>;

/// The five SDK handlers a display keeps.
#[derive(Default)]
struct Handlers {
    mode_change: Option<ModeChangeHandler>,
    new_frame_event: Option<NewFrameEventHandler>,
    cursor_glyph: Option<CursorGlyphHandler>,
    cursor_show: Option<CursorShowHandler>,
    cursor_move: Option<CursorMoveHandler>,
    sleep: Option<SleepHandler>,
}

/// The SDK's readonly properties a display keeps (the size, the queue, the five handlers, the
/// port, the serial number, the present counts), taken from the descriptor at init. The queue
/// and the handlers are released where the unplug and the end release them.
pub(super) struct SdkProps {
    size_in_millimeters: (f64, f64),
    port: u64,
    serial_num: u32,
    guest_present_count: AtomicU64,
    host_present_count: AtomicU64,
    /// The display's own reference to the descriptor's queue, until the unplug.
    queue: Mutex<Option<Arc<dyn HostQueue>>>,
    handlers: Mutex<Handlers>,
}

impl SdkProps {
    /// The properties `desc` holds, which it gives up (the handlers are the same blocks). The
    /// queue is `desc`'s reference until the initializer takes the display's own
    /// ([`SdkProps::retain_queue`]).
    pub(super) fn take(desc: &mut DisplayDescriptor, port: u64, serial_num: u32) -> SdkProps {
        SdkProps {
            size_in_millimeters: std::mem::take(&mut desc.size_in_millimeters),
            port,
            serial_num,
            guest_present_count: AtomicU64::new(0),
            host_present_count: AtomicU64::new(0),
            queue: Mutex::new(desc.queue.take()),
            handlers: Mutex::new(Handlers {
                mode_change: desc.mode_change_handler.take(),
                new_frame_event: desc.new_frame_event_handler.take(),
                cursor_glyph: desc.cursor_glyph_handler.take(),
                cursor_show: desc.cursor_show_handler.take(),
                cursor_move: desc.cursor_move_handler.take(),
                sleep: desc.sleep_handler.take(),
            }),
        }
    }

    pub(super) fn size_in_millimeters(&self) -> (f64, f64) {
        self.size_in_millimeters
    }

    pub(super) fn port(&self) -> u64 {
        self.port
    }

    pub(super) fn serial_num(&self) -> u32 {
        self.serial_num
    }

    pub(super) fn guest_present_count(&self) -> u64 {
        self.guest_present_count.load(Ordering::SeqCst)
    }

    pub(super) fn set_guest_present_count(&self, count: u64) {
        self.guest_present_count.store(count, Ordering::SeqCst);
    }

    pub(super) fn host_present_count(&self) -> u64 {
        self.host_present_count.load(Ordering::SeqCst)
    }

    pub(super) fn set_host_present_count(&self, count: u64) {
        self.host_present_count.store(count, Ordering::SeqCst);
    }

    pub(super) fn queue(&self) -> Option<Arc<dyn HostQueue>> {
        self.queue.lock().clone()
    }

    /// The display's own `dispatch_retain` of the queue, in place of the descriptor's.
    pub(super) fn retain_queue(&self) {
        let shared = self.queue.lock().clone();
        if let Some(q) = shared {
            let own = q.dispatch_retain();
            let old = self.queue.lock().replace(own);
            drop(old);
        }
    }

    /// Releases the display's queue.
    pub(super) fn release_queue(&self) {
        let q = self.queue.lock().take();
        // Outside the lock: the last release may run the host's code.
        drop(q);
    }

    pub(super) fn new_frame_event_handler(&self) -> Option<NewFrameEventHandler> {
        self.handlers.lock().new_frame_event.clone()
    }

    /// Releases the five handlers, in a fixed order, each taken under the lock and dropped after
    /// it: a last release runs the host's destroy.
    pub(super) fn release_handlers(&self) {
        let h = std::mem::take(&mut self.handlers.lock().mode_change);
        drop(h);
        let h = std::mem::take(&mut self.handlers.lock().new_frame_event);
        drop(h);
        let h = std::mem::take(&mut self.handlers.lock().cursor_glyph);
        drop(h);
        let h = std::mem::take(&mut self.handlers.lock().cursor_show);
        drop(h);
        let h = std::mem::take(&mut self.handlers.lock().cursor_move);
        drop(h);
        let h = std::mem::take(&mut self.handlers.lock().sleep);
        drop(h);
    }
}

impl<P: Platform> DisplayCore<P> {
    /// The mode change, taking the display mutex (tests; the product's present holds it).
    #[cfg(test)]
    pub(crate) fn mode_change(
        &self,
        width: u64,
        height: u64,
        iosurface_pixel_format: u32,
        protection_requirements: u64,
    ) {
        let mut g = self.display_mutex.lock();
        self.mode_change_locked(
            &mut g,
            width,
            height,
            iosurface_pixel_format,
            protection_requirements,
        );
    }

    /// The mode change, with the display mutex held: logs, stores the protection requirements,
    /// dispatches the mode change handler onto the queue (it reads the handler when it runs),
    /// then stores the size and the format.
    pub(super) fn mode_change_locked(
        &self,
        g: &mut DisplayLocked<P>,
        width: u64,
        height: u64,
        iosurface_pixel_format: u32,
        protection_requirements: u64,
    ) {
        log::info!(
            "display on port {}: mode {width}x{height}, pixel format {iosurface_pixel_format:#x}, \
             protection requirements {protection_requirements:#x}",
            self.port()
        );
        g.protection_requirements = protection_requirements;
        self.protection_requirements_unlocked
            .store(protection_requirements, Ordering::SeqCst);
        if self.mode_change_handler().is_some() {
            // The present asserted the queue, so only an unplug since can have cleared it.
            let Some(queue) = self.queue() else {
                crate::fatal!(DeathKind::ModeChangeWithoutQueue)
            };
            // A display whose end waits in the detach still dispatches it.
            if let Some(this) = self.block_self() {
                let block = move || {
                    // The handler read when the block runs; the end's release of it since is a
                    // call through nil.
                    match this.mode_change_handler() {
                        Some(h) => h(
                            DisplayCoord {
                                x: width.low_u16(),
                                y: height.low_u16(),
                            },
                            iosurface_pixel_format,
                        ),
                        None => crate::fatal!(DeathKind::ModeChangeHandlerGone),
                    }
                };
                queue.exec_async(Box::new(block));
            }
        }
        g.width = width.low_u16();
        g.height = height.low_u16();
        g.iosurface_pixel_format = iosurface_pixel_format;
    }

    /// The mode change handler.
    pub fn mode_change_handler(&self) -> Option<ModeChangeHandler> {
        self.props.handlers.lock().mode_change.clone()
    }

    /// The newFrame handler.
    pub fn new_frame_event_handler(&self) -> Option<NewFrameEventHandler> {
        self.props.new_frame_event_handler()
    }

    /// The cursor glyph handler.
    pub fn cursor_glyph_handler(&self) -> Option<CursorGlyphHandler> {
        self.props.handlers.lock().cursor_glyph.clone()
    }

    /// The cursor show handler.
    pub fn cursor_show_handler(&self) -> Option<CursorShowHandler> {
        self.props.handlers.lock().cursor_show.clone()
    }

    /// The cursor move handler.
    pub fn cursor_move_handler(&self) -> Option<CursorMoveHandler> {
        self.props.handlers.lock().cursor_move.clone()
    }

    /// The sleep handler (not an SDK handler).
    pub fn sleep_handler(&self) -> Option<SleepHandler> {
        self.props.handlers.lock().sleep.clone()
    }
}
