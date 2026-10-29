//! The display descriptor: what a new display copies. [`DisplayDescriptor::sdk_defaults`] gives
//! the SDK's defaults; the derived `Default` is all zero and nil, which is not (no name, a zero colour
//! space, origin {0, 0}).
//!
//! The name is a copy of the host's text; the guest display reads the display's copy at every
//! settings update. The private properties whose values the guest display writes into the page keep
//! the SDK's defaults; the private handlers are not here, as no front end sets them.

use std::sync::Arc;

use crate::display::host_calls::{
    CursorGlyphHandler, CursorMoveHandler, CursorShowHandler, GlyphImageMaker, ModeChangeHandler,
    NewFrameEventHandler, SleepHandler,
};
use crate::display::{DisplayChromaticity, DisplayColorSpace, DisplayCoord, DisplayName};
use crate::host::HostQueue;

/// The default name.
pub const DEFAULT_NAME: &str = "Apple Virtual";

/// The default colour space: red (0.64, 0.33), green (0.30, 0.60), blue (0.15, 0.06), white
/// (0.312727, 0.329023), as float literals.
pub const DEFAULT_COLOR_SPACE: DisplayColorSpace = DisplayColorSpace {
    red_primary: DisplayChromaticity { x: 0.64, y: 0.33 },
    green_primary: DisplayChromaticity { x: 0.30, y: 0.60 },
    blue_primary: DisplayChromaticity { x: 0.15, y: 0.06 },
    white_point: DisplayChromaticity {
        x: 0.312_727,
        y: 0.329_023,
    },
};

/// The default origin: {0xffff, 0xffff}.
pub const DEFAULT_ORIGIN: DisplayCoord = DisplayCoord {
    x: 0xffff,
    y: 0xffff,
};

/// What a new display copies: one field per property of the SDK's display descriptor, its
/// private ones included, named after it. The public APIs fill it from their display
/// configuration, with the display thread as its queue. `None` is a nil handler or property.
#[derive(Clone, Default)]
pub struct DisplayDescriptor {
    /// `name`: a copy of the host's text.
    pub name: Option<DisplayName>,
    /// `sizeInMillimeters` (`CGSize`).
    pub size_in_millimeters: (f64, f64),
    /// `queue`: nil by default; nothing substitutes the main queue.
    pub queue: Option<Arc<dyn HostQueue>>,
    /// `modeChangeHandler`.
    pub mode_change_handler: Option<ModeChangeHandler>,
    /// `newFrameEventHandler`.
    pub new_frame_event_handler: Option<NewFrameEventHandler>,
    /// `cursorGlyphHandler`.
    pub cursor_glyph_handler: Option<CursorGlyphHandler>,
    /// Not an SDK property: the host's maker of the image the guest display builds for
    /// `cursorGlyphHandler`, called on the device thread. `None`, which the public APIs give,
    /// hands the handler the pixels alone.
    pub cursor_glyph_image_maker: Option<GlyphImageMaker>,
    /// `cursorShowHandler`.
    pub cursor_show_handler: Option<CursorShowHandler>,
    /// `cursorMoveHandler`.
    pub cursor_move_handler: Option<CursorMoveHandler>,
    /// Not an SDK property: told the guest display's sleep and wake, on the display's queue.
    pub sleep_handler: Option<SleepHandler>,
    /// `colorSpace`.
    pub color_space: DisplayColorSpace,
    /// `configEpoch`.
    pub config_epoch: u64,
    /// `origin`.
    pub origin: DisplayCoord,
    /// `scaleFactor`.
    pub scale_factor: u8,
    /// `connectionType`.
    pub connection_type: i32,
    /// `displayUUID`.
    pub display_uuid: Option<String>,
    /// `minNits`.
    pub min_nits: f32,
    /// `maxNits`.
    pub max_nits: f32,
    /// `maxSDRNits`.
    pub max_sdr_nits: f32,
}

impl DisplayDescriptor {
    /// The SDK's defaults: the name "Apple Virtual", the default colour space, origin
    /// {0xffff, 0xffff}, everything else zero or nil (the queue too: nothing falls back to the
    /// main queue). The nits are zero: no preference is read.
    pub fn sdk_defaults() -> DisplayDescriptor {
        DisplayDescriptor {
            name: Some(DisplayName::new(DEFAULT_NAME)),
            color_space: DEFAULT_COLOR_SPACE,
            origin: DEFAULT_ORIGIN,
            ..DisplayDescriptor::default()
        }
    }
}

impl std::fmt::Debug for DisplayDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DisplayDescriptor")
            .field("name", &self.name)
            .field("size_in_millimeters", &self.size_in_millimeters)
            .field("port_queue", &self.queue.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod sdk_defaults {
        use super::*;

        /// The nits are zero: nothing overrides them.
        #[test]
        fn nits_are_zero() {
            let d = DisplayDescriptor::sdk_defaults();
            assert_eq!((d.min_nits, d.max_nits, d.max_sdr_nits), (0.0, 0.0, 0.0));
            assert_eq!(d.origin, DEFAULT_ORIGIN);
        }

        /// The SDK's defaults, each value exact.
        #[test]
        fn defaults() {
            let d = DisplayDescriptor::sdk_defaults();
            assert_eq!(
                d.name.as_ref().map(DisplayName::current).as_deref(),
                Some("Apple Virtual")
            );
            assert_eq!(
                d.color_space.red_primary,
                DisplayChromaticity { x: 0.64, y: 0.33 }
            );
            assert_eq!(
                d.color_space.green_primary,
                DisplayChromaticity { x: 0.30, y: 0.60 }
            );
            assert_eq!(
                d.color_space.blue_primary,
                DisplayChromaticity { x: 0.15, y: 0.06 }
            );
            assert_eq!(
                d.color_space.white_point,
                DisplayChromaticity {
                    x: 0.312727,
                    y: 0.329023
                }
            );
            // The float literals' exact bits.
            assert_eq!(d.color_space.white_point.x.to_bits(), 0.312727f32.to_bits());
            assert_eq!(
                d.origin,
                DisplayCoord {
                    x: 0xffff,
                    y: 0xffff
                }
            );
            assert_eq!(d.size_in_millimeters, (0.0, 0.0));
            assert!(d.queue.is_none());
            assert!(d.mode_change_handler.is_none());
            assert!(d.new_frame_event_handler.is_none());
            assert!(d.cursor_glyph_handler.is_none());
            assert!(d.cursor_show_handler.is_none());
            assert!(d.cursor_move_handler.is_none());
            assert_eq!(d.config_epoch, 0);
            assert_eq!(d.scale_factor, 0);
            assert_eq!(d.connection_type, 0);
            assert!(d.display_uuid.is_none());
            assert_eq!((d.min_nits, d.max_nits, d.max_sdr_nits), (0.0, 0.0, 0.0));
        }

        /// `Default` is not the SDK's init (the module doc): no name, zero colour space and
        /// origin.
        #[test]
        fn default_is_not_init() {
            let d = DisplayDescriptor::default();
            assert!(d.name.is_none());
            assert_eq!(d.origin, DisplayCoord::default());
            assert_eq!(d.color_space, DisplayColorSpace::default());
        }
    }

    /// The last drop releases what the descriptor holds
    #[test]
    fn drop_releases_handlers() {
        let mut d = DisplayDescriptor::sdk_defaults();
        let h: NewFrameEventHandler = Arc::new(|| {});
        let weak = Arc::downgrade(&h);
        d.new_frame_event_handler = Some(h);
        d.name = Some("X".into());
        d.display_uuid = Some("U".into());
        let copy = d.clone();
        drop(d);
        assert!(weak.upgrade().is_some());
        drop(copy);
        assert!(weak.upgrade().is_none());
    }

    /// The Debug text names the port queue by presence only
    #[test]
    fn debug_text() {
        let d = DisplayDescriptor::sdk_defaults();
        let s = format!("{d:?}");
        assert!(s.contains("Apple Virtual") && s.contains("port_queue: false"));
    }
}
