//! The display state page a guest display shares with the guest (one guest page): each field's
//! offset and width.
//!
//! The page is guest memory (D18): the guest display reaches each field through
//! `GuestPage::at(OFF)` with a volatile or atomic access of the field's width, never through a Rust
//! reference, so the guest sees the raw bytes the guest display writes, in the order it writes
//! them. The field names are ours, after what the code does with each; only `modeList.count` and
//! `kAppleParavirtGPUDisplayMaxModes` are stated by the image. A field nothing names stays
//! `FIELD_XX`. Ranges nothing touches have no constant. All multi-byte fields are little-endian.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::guest::records::APV_PAGE_SIZE;

/// The page's length: the guest display maps one guest page from the page's number, 16 KiB, and
/// 4 KiB with `cfg(apv_guest_4k)`. Every field ends by 0xfd8, inside either. The colour matrices at the
/// guest's index reach 0x3a10 ([`color_matrix`]): inside a 16 KiB page, and past a 4 KiB one
/// from index 31, where each access to them is checked ([`color_matrix_in_page`]).
pub const PAGE_LENGTH: u64 = APV_PAGE_SIZE;

/// `kAppleParavirtGPUDisplayMaxModes` — the one constant the image names (the assert in
/// `-testUpdateModeList:` spells it).
pub const MAX_MODES: u16 = 128;

// ---- PGDisplaySharedStateLayout ---------------------------------

/// `serialNum`: u32, host-written (`_PGDisplay.serialNum`).
pub const SERIAL_NUM: u64 = 0x000;
/// `name`: `char[14]`, host-written (`_PGDisplay.name`).
pub const NAME: u64 = 0x004;
/// The length of `name`.
pub const NAME_LEN: u64 = 14;
/// `port`: u16, the guest display's port, written at guest display init only.
pub const PORT: u64 = 0x012;
/// `roundedWidthInMillimeters`: u16.
pub const ROUNDED_WIDTH_IN_MILLIMETERS: u64 = 0x014;
/// `roundedHeightInMillimeters`: u16.
pub const ROUNDED_HEIGHT_IN_MILLIMETERS: u64 = 0x016;
/// `cursorGlyphMaxWidth`: u16 (256 with a glyph handler).
pub const CURSOR_GLYPH_MAX_WIDTH: u64 = 0x018;
/// `cursorGlyphMaxHeight`: u16.
pub const CURSOR_GLYPH_MAX_HEIGHT: u64 = 0x01a;
/// `field1c`: u32, set to 1024 at guest display init and never read.
pub const FIELD_1C: u64 = 0x01c;
/// `cursorHandlers`: u32, bit 0 show, bit 1 move.
pub const CURSOR_HANDLERS: u64 = 0x020;
/// `widthInMillimeters`: f32.
pub const WIDTH_IN_MILLIMETERS: u64 = 0x024;
/// `heightInMillimeters`: f32.
pub const HEIGHT_IN_MILLIMETERS: u64 = 0x028;
/// `colorSpace`: `PGDisplayColorSpace`, eight f32s (red, green, blue, white; x then y).
pub const COLOR_SPACE: u64 = 0x02c;
/// `origin`: `PGDisplayCoord_t` (u16 x, u16 y).
pub const ORIGIN: u64 = 0x04c;
/// `scaleFactor`: u8.
pub const SCALE_FACTOR: u64 = 0x050;
/// `compositorSupportsLiveResize`: u8, guest-written, read by `-doorbell` (nonzero = set).
pub const COMPOSITOR_SUPPORTS_LIVE_RESIZE: u64 = 0x051;
/// `configEpoch`: u64.
pub const CONFIG_EPOCH: u64 = 0x058;
/// `connectionType`: i32.
pub const CONNECTION_TYPE: u64 = 0x060;
/// `protectionOptions`: u64.
pub const PROTECTION_OPTIONS: u64 = 0x068;
/// `minNits`: f32.
pub const MIN_NITS: u64 = 0x070;
/// `maxNits`: f32.
pub const MAX_NITS: u64 = 0x074;
/// `maxSDRNits`: f32.
pub const MAX_SDR_NITS: u64 = 0x078;
/// `raisedInterrupts`: `std::atomic<uint32_t>` (the host ORs, the guest clears).
pub const RAISED_INTERRUPTS: u64 = 0x100;
/// `enabledInterrupts`: `std::atomic<uint32_t>`, guest-written.
pub const ENABLED_INTERRUPTS: u64 = 0x104;
/// `expectedAck`: u32.
pub const EXPECTED_ACK: u64 = 0x200;
/// `modeList.count`: u16 (a name the image states).
pub const MODE_LIST_COUNT: u64 = 0x208;
/// `modeList.modes`: `MAX_MODES` entries of `MODE_ENTRY_SIZE` bytes.
pub const MODE_LIST_MODES: u64 = 0x210;
/// `colorMatrices`: `float[2][3][4]`, row-major.
pub const COLOR_MATRICES: u64 = 0xa10;
/// `colorMatrixReadIndex`: u8, read with a volatile load.
pub const COLOR_MATRIX_READ_INDEX: u64 = 0xa70;
/// `colorMatrixForceRead`: u8.
pub const COLOR_MATRIX_FORCE_READ: u64 = 0xa71;
/// `colorMatrixWriteIndex`: u8, guest-written.
pub const COLOR_MATRIX_WRITE_INDEX: u64 = 0xa72;
/// `colorMatrixDirty`: u8.
pub const COLOR_MATRIX_DIRTY: u64 = 0xa73;
/// `displayUUID`: `char[39]`.
pub const DISPLAY_UUID: u64 = 0xa74;
/// The length of `displayUUID`.
pub const DISPLAY_UUID_LEN: u64 = 39;
/// `cursorPosition`: `std::atomic<uint32_t>`, x in the low half, y in the high half.
pub const CURSOR_POSITION: u64 = 0xe00;
/// `guestCursorShow`: the guest's raw byte.
pub const GUEST_CURSOR_SHOW: u64 = 0xe04;
/// `cursorGlyphVirtualOffset`: u64.
pub const CURSOR_GLYPH_VIRTUAL_OFFSET: u64 = 0xf00;
/// `cursorGlyphMappedLength`: u64.
pub const CURSOR_GLYPH_MAPPED_LENGTH: u64 = 0xf08;
/// `cursorGlyphStride`: u64.
pub const CURSOR_GLYPH_STRIDE: u64 = 0xf10;
/// `fieldf18`: u32, set to 0 with each glyph.
pub const FIELD_F18: u64 = 0xf18;
/// `cursorGlyphSum`: u32.
pub const CURSOR_GLYPH_SUM: u64 = 0xf1c;
/// `cursorShow`: the raw byte, host side.
pub const CURSOR_SHOW: u64 = 0xf20;
/// `cursorGlyphWidth`: u16.
pub const CURSOR_GLYPH_WIDTH: u64 = 0xf22;
/// `cursorGlyphHeight`: u16.
pub const CURSOR_GLYPH_HEIGHT: u64 = 0xf24;
/// `cursorGlyphHotSpotX`: u16.
pub const CURSOR_GLYPH_HOT_SPOT_X: u64 = 0xf26;
/// `cursorGlyphHotSpotY`: u16.
pub const CURSOR_GLYPH_HOT_SPOT_Y: u64 = 0xf28;
/// `gammaTableVirtualOffset`: u64.
pub const GAMMA_TABLE_VIRTUAL_OFFSET: u64 = 0xf80;
/// `gammaTableMappedLength`: u64.
pub const GAMMA_TABLE_MAPPED_LENGTH: u64 = 0xf88;
/// `gammaTableEntryCount`: u32.
pub const GAMMA_TABLE_ENTRY_COUNT: u64 = 0xf90;
/// `gammaTableSum`: u32.
pub const GAMMA_TABLE_SUM: u64 = 0xf94;
/// `presentCount`: u64.
pub const PRESENT_COUNT: u64 = 0xf98;
/// `presentedSurface`: u32.
pub const PRESENTED_SURFACE: u64 = 0xfa0;
/// `fieldfa4`: u32, set to 0 by each present.
pub const FIELD_FA4: u64 = 0xfa4;
/// `ackPending`: a byte; `-ack:` tests bit 0, the state machine `== 1`.
pub const ACK_PENDING: u64 = 0xfa9;
/// `presentedMappingID`: u64.
pub const PRESENTED_MAPPING_ID: u64 = 0xfb0;
/// `asleep`: the raw byte.
pub const ASLEEP: u64 = 0xfb8;
/// `lastAck`: u32.
pub const LAST_ACK: u64 = 0xfbc;
/// `state`: u32 (0 disconnected, 1 awaiting ack, 2 connected).
pub const STATE: u64 = 0xfc0;
/// `guestICCProfileVirtualOffset`: u64.
pub const GUEST_ICC_PROFILE_VIRTUAL_OFFSET: u64 = 0xfc8;
/// `guestICCProfileLength`: u32.
pub const GUEST_ICC_PROFILE_LENGTH: u64 = 0xfd0;

// ---- PGDisplaySharedStateMode ------------------------------------

/// `sizeof(PGDisplaySharedStateMode)`.
pub const MODE_ENTRY_SIZE: u64 = 16;
/// `width`: u16 within an entry.
pub const MODE_WIDTH: u64 = 0;
/// `height`: u16 within an entry.
pub const MODE_HEIGHT: u64 = 2;
/// `refreshRate`: u32 within an entry, 16.16 fixed point.
pub const MODE_REFRESH_RATE: u64 = 4;
/// The flags byte within an entry: bit 0 `supportsEDR`, bit 1 `supports10bpc`, bits 2-7 kept
/// (a read-modify-write of the byte).
pub const MODE_FLAGS: u64 = 8;
/// `supportsEDR`'s bit in `MODE_FLAGS`.
pub const MODE_FLAG_SUPPORTS_EDR: u8 = 1 << 0;
/// `supports10bpc`'s bit in `MODE_FLAGS`.
pub const MODE_FLAG_SUPPORTS_10BPC: u8 = 1 << 1;

/// The size of one colour matrix: `float[3][4]`.
pub const COLOR_MATRIX_SIZE: u64 = 48;

/// The page offset of `modeList.modes[index]`. The guest display indexes with the guest's or the
/// host's count unchecked where it does (`index` is at most 0xffff, so the offset stays in a u64);
/// whether it is in bounds is the caller's to decide.
pub fn mode_entry(index: u16) -> u64 {
    // Bounded: 0x210 + 0xffff * 16 < 2^64.
    MODE_LIST_MODES.wrapping_add(u64::from(index).wrapping_mul(MODE_ENTRY_SIZE))
}

/// The page offset of `colorMatrices[matrix][row][col]`. `matrix` is a guest byte (0-255: at most
/// 0xa10 + 255 * 48 + 48 = 0x3a10, inside a 16 KiB page, past a 4 KiB one from index 31);
/// `row` < 3 and `col` < 4 are the matrix's own bounds.
pub fn color_matrix(matrix: u8, row: u8, col: u8) -> u64 {
    let m = u64::from(matrix).wrapping_mul(COLOR_MATRIX_SIZE);
    let r = u64::from(row).wrapping_mul(16);
    let c = u64::from(col).wrapping_mul(4);
    COLOR_MATRICES
        .wrapping_add(m)
        .wrapping_add(r.wrapping_add(c))
}

/// Whether the `width` bytes at `colorMatrices[matrix][row][col]` lie inside the page. Always on
/// 16 KiB guest pages, where matrix 255 ends at 0x3a10; on 4 KiB ones every access of matrices
/// 0-30 (matrix 30 ends at 0xfe0), and of matrix 31 those that end by 0x1000 (its rows 0 and 1).
#[inline]
pub fn color_matrix_in_page(matrix: u8, row: u8, col: u8, width: u64) -> bool {
    color_matrix(matrix, row, col)
        .checked_add(width)
        .is_some_and(|end| end <= PAGE_LENGTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_offsets_hold() {
        assert_eq!(PORT, 0x12);
        assert_eq!(COLOR_SPACE, 0x2c);
        assert_eq!(CONFIG_EPOCH, 0x58);
        assert_eq!(PROTECTION_OPTIONS, 0x68);
        assert_eq!(RAISED_INTERRUPTS, 0x100);
        assert_eq!(EXPECTED_ACK, 0x200);
        assert_eq!(MODE_LIST_COUNT, 0x208);
        assert_eq!(COLOR_MATRICES, 0xa10);
        assert_eq!(DISPLAY_UUID, 0xa74);
        assert_eq!(CURSOR_POSITION, 0xe00);
        assert_eq!(CURSOR_GLYPH_VIRTUAL_OFFSET, 0xf00);
        assert_eq!(CURSOR_GLYPH_WIDTH, 0xf22);
        assert_eq!(GAMMA_TABLE_VIRTUAL_OFFSET, 0xf80);
        assert_eq!(ACK_PENDING, 0xfa9);
        assert_eq!(PRESENTED_MAPPING_ID, 0xfb0);
        assert_eq!(STATE, 0xfc0);
        assert_eq!(GUEST_ICC_PROFILE_VIRTUAL_OFFSET, 0xfc8);
        assert_eq!(GUEST_ICC_PROFILE_LENGTH, 0xfd0);
        assert_eq!(MODE_ENTRY_SIZE, 16);
        assert_eq!(PAGE_LENGTH, crate::guest::records::APV_PAGE_SIZE);
        #[cfg(not(apv_guest_4k))]
        assert_eq!(PAGE_LENGTH, 0x4000);
        #[cfg(apv_guest_4k)]
        assert_eq!(PAGE_LENGTH, 0x1000);
    }

    /// The fields that follow a fixed-size array end where the next begins
    #[test]
    fn array_fields_abut() {
        assert_eq!(NAME + NAME_LEN, PORT);
        assert_eq!(COLOR_SPACE + 8 * 4, ORIGIN);
        assert_eq!(mode_entry(MAX_MODES), COLOR_MATRICES);
        assert_eq!(color_matrix(2, 0, 0), COLOR_MATRIX_READ_INDEX);
        assert_eq!(DISPLAY_UUID + DISPLAY_UUID_LEN, 0xa9b);
    }

    /// modeList.modes[i] and the fields of an entry
    #[test]
    fn mode_entry_offsets() {
        assert_eq!(mode_entry(0), 0x210);
        assert_eq!(mode_entry(1), 0x220);
        assert_eq!(mode_entry(127), 0xa00);
        // Past the array (an overrun, if a caller makes one): no wrap in u64.
        assert_eq!(mode_entry(0xffff), 0x210 + 0xffff * 16);
        assert_eq!(
            (MODE_WIDTH, MODE_HEIGHT, MODE_REFRESH_RATE, MODE_FLAGS),
            (0, 2, 4, 8)
        );
        assert_eq!(MODE_FLAG_SUPPORTS_EDR | MODE_FLAG_SUPPORTS_10BPC, 0b11);
    }

    /// colorMatrices[w][r][c]: w up to 255 reaches 0x3a10
    #[test]
    fn color_matrix_offsets() {
        assert_eq!(color_matrix(0, 0, 0), 0xa10);
        assert_eq!(color_matrix(0, 1, 1), 0xa10 + 16 + 4);
        assert_eq!(color_matrix(0, 2, 2), 0xa10 + 32 + 8);
        assert_eq!(color_matrix(1, 0, 0), 0xa40);
        assert_eq!(color_matrix(255, 2, 3) + 4, 0x3a10);
    }

    /// A 16 KiB page holds every matrix: every access is inside
    #[cfg(not(apv_guest_4k))]
    #[test]
    fn every_color_matrix_is_in_a_16k_page() {
        assert!(color_matrix(255, 2, 3) + 4 <= PAGE_LENGTH);
        for w in 0..=255u8 {
            assert!(color_matrix_in_page(w, 0, 0, COLOR_MATRIX_SIZE), "{w}");
            assert!(color_matrix_in_page(w, 2, 3, 4), "{w}");
        }
    }

    /// A 4 KiB page holds matrices 0-30 whole and the first 32 bytes of matrix 31 (its rows 0
    /// and 1); every access past those is outside
    #[cfg(apv_guest_4k)]
    #[test]
    fn a_4k_page_ends_in_matrix_31() {
        assert_eq!(color_matrix(30, 2, 3) + 4, 0xfe0);
        assert_eq!(color_matrix(31, 0, 0), 0xfe0);
        for w in 0..=30u8 {
            assert!(color_matrix_in_page(w, 0, 0, COLOR_MATRIX_SIZE), "{w}");
            assert!(color_matrix_in_page(w, 2, 3, 4), "{w}");
        }
        assert!(color_matrix_in_page(31, 0, 0, 4));
        assert!(color_matrix_in_page(31, 1, 1, 4));
        assert!(color_matrix_in_page(31, 1, 2, 8));
        assert!(color_matrix_in_page(31, 1, 3, 4));
        assert!(!color_matrix_in_page(31, 2, 0, 4));
        assert!(!color_matrix_in_page(31, 2, 2, 4));
        assert!(!color_matrix_in_page(31, 0, 0, COLOR_MATRIX_SIZE));
        for w in 32..=255u8 {
            assert!(!color_matrix_in_page(w, 0, 0, 4), "{w}");
        }
    }
}
