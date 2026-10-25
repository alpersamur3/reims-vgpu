//! The guest ABI's records: `APVFeatures` and the records a guest writes, `#[repr(C)]` with
//! their sizes asserted.
//!
//! The names are the guest ABI's (docs/testing.md section 6), or, where a doc says so, named after
//! what their readers do; a field nothing names is `fieldNN`, its byte offset in hex. A guest
//! record is only ever copied out of guest memory as bytes (`guest::ptr`) and read from the private
//! copy, never referenced in place.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::num::Truncate;

/// The guest page shift: 14 for an arm64 guest's 16 KiB pages, 12 for an x86-64 guest's 4 KiB
/// ones. The guest is the build's architecture unless the `guest-arm64` feature makes it arm64:
/// build.rs sets `cfg(apv_guest_4k)` only for an x86-64 build without it (core's tests also run
/// as x86-64 under Rosetta on a Mac). Every page number the guest hands the device counts these
/// pages; the host VM's page (`HostVm::page_size`) may be smaller. Name is ours: the shift of
/// `APV_PAGE_SIZE`.
pub const APV_PAGE_SHIFT: u32 = if cfg!(apv_guest_4k) { 12 } else { 14 };

/// `APV_PAGE_SIZE`: the guest page size, 16 KiB (4 KiB with `cfg(apv_guest_4k)`).
pub const APV_PAGE_SIZE: u64 = 1_u64.wrapping_shl(APV_PAGE_SHIFT);

/// A guest address's offset in its page. Name is ours: the mask of `APV_PAGE_SIZE`.
pub const APV_PAGE_MASK: u64 = APV_PAGE_SIZE.wrapping_sub(1);

/// `APV_MAX_FIFOS`: the FIFOs a guest may stamp through, the root included.
pub const APV_MAX_FIFOS: u32 = 64;

/// `APVBinaryVersion`: compared signed and 32 bits wide; no enumerator is attested.
pub type BinaryVersion = i32;

/// `APVFeatures`, the features the guest's binary version turns on: every field name and type is
/// the metadata's, and the type's name is ours. 60 bytes; the device owns one, each isolated task's
/// service a copy (the handshake sends it by value).
///
/// `BOOL` is `bool` on arm64, so the host-owned copy holds `bool`s. It never lives in guest memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
#[allow(missing_docs)] // the metadata's field names
pub struct GuestFeatures {
    pub valid: bool,
    pub unsupported: bool,
    pub object_tables: bool,
    pub efi_display: bool,
    pub display_doorbell: bool,
    pub mapper_io_surfaces: bool,
    pub synchronize_and_discard_resources: bool,
    pub display_mapper_surface: bool,
    pub discard_resources: bool,
    pub display_sleep: bool,
    pub supports_rgha_presents: bool,
    pub synchronize_discard_child_resources: bool,
    pub dual_plane_textures: bool,
    pub display_set_properties: bool,
    pub display_dimension_floats: bool,
    pub metal_heaps: bool,
    pub buffer_from_io_surface: bool,
    pub max_fifo_count: u32,
    pub s8_byte_count_for_d32s8: u32,
    pub supports_shared_textures: bool,
    pub supports_info_encoder: bool,
    pub supports_default_raster_sample_count: bool,
    pub supports_vertex_amplification: bool,
    pub supports_sunburst_display: bool,
    pub supports_programmable_sample_positions: bool,
    pub supports_variable_rasterization_rate_maps: bool,
    pub supports_tile_shaders_image_blocks: bool,
    pub supports_display_compositor_parameters: bool,
    pub supports_cmd_exec_indirect3: bool,
    pub supports_swizzled_textures: bool,
    pub supports_dynamic_attribute_stride: bool,
    pub supports_blit_encoder_spi: bool,
    pub supports_multiplane_mipped_blit_textures: bool,
    pub supports_compute_pass_descriptor_dispatch_type: bool,
    pub supports_object_unique_identifier: bool,
    pub supports_command_buffer_jump: bool,
    pub supports_shared_storage_heaps: bool,
    pub supports_heap_hazard_tracking: bool,
    pub supports_protection_options_envelope: bool,
    pub supports_insert_compressed_texture_reinterpretation_flush: bool,
    pub supports_argument_buffers: bool,
    pub supports_display_set_guest_icc_profile: bool,
    pub supports_correct_2p_xr10_a8: bool,
    pub supports_cpu_buffers: bool,
    pub supports_indirect_command_buffers: bool,
    pub supports_protected_texture_options: bool,
    pub supports_dispatch_threads_indirect: bool,
    pub supports_info_indirect: bool,
    pub supports_display_port_connection_type: bool,
    pub supports_texture_descriptor2: bool,
    pub supports_multiple_displays: bool,
}

const _: () = assert!(size_of::<GuestFeatures>() == 60);
const _: () = assert!(std::mem::offset_of!(GuestFeatures, max_fifo_count) == 20);
const _: () = assert!(std::mem::offset_of!(GuestFeatures, supports_multiple_displays) == 59);

/// `APVFeatures` as its bytes: one `u8` per `BOOL`, the three padding bytes before
/// `maxFIFOCount` spelled out, and the conversions to and from [`GuestFeatures`]. One field list
/// feeds all three, so the order cannot drift from the struct's.
macro_rules! features_layout {
    (head: [$($h:ident),* $(,)?], tail: [$($t:ident),* $(,)?] $(,)?) => {
        /// The ABI's layout of `APVFeatures`, byte for byte.
        #[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
        #[repr(C)]
        struct RawFeatures {
            $($h: u8,)*
            padding: [u8; 3],
            max_fifo_count: u32,
            s8_byte_count_for_d32s8: u32,
            $($t: u8,)*
        }

        impl RawFeatures {
            fn of(f: &GuestFeatures) -> RawFeatures {
                RawFeatures {
                    $($h: u8::from(f.$h),)*
                    padding: [0; 3],
                    max_fifo_count: f.max_fifo_count,
                    s8_byte_count_for_d32s8: f.s8_byte_count_for_d32s8,
                    $($t: u8::from(f.$t),)*
                }
            }

            fn features(&self) -> GuestFeatures {
                GuestFeatures {
                    $($h: (self.$h != 0),)*
                    max_fifo_count: self.max_fifo_count,
                    s8_byte_count_for_d32s8: self.s8_byte_count_for_d32s8,
                    $($t: (self.$t != 0),)*
                }
            }
        }
    };
}

features_layout! {
    head: [
        valid,
        unsupported,
        object_tables,
        efi_display,
        display_doorbell,
        mapper_io_surfaces,
        synchronize_and_discard_resources,
        display_mapper_surface,
        discard_resources,
        display_sleep,
        supports_rgha_presents,
        synchronize_discard_child_resources,
        dual_plane_textures,
        display_set_properties,
        display_dimension_floats,
        metal_heaps,
        buffer_from_io_surface,
    ],
    tail: [
        supports_shared_textures,
        supports_info_encoder,
        supports_default_raster_sample_count,
        supports_vertex_amplification,
        supports_sunburst_display,
        supports_programmable_sample_positions,
        supports_variable_rasterization_rate_maps,
        supports_tile_shaders_image_blocks,
        supports_display_compositor_parameters,
        supports_cmd_exec_indirect3,
        supports_swizzled_textures,
        supports_dynamic_attribute_stride,
        supports_blit_encoder_spi,
        supports_multiplane_mipped_blit_textures,
        supports_compute_pass_descriptor_dispatch_type,
        supports_object_unique_identifier,
        supports_command_buffer_jump,
        supports_shared_storage_heaps,
        supports_heap_hazard_tracking,
        supports_protection_options_envelope,
        supports_insert_compressed_texture_reinterpretation_flush,
        supports_argument_buffers,
        supports_display_set_guest_icc_profile,
        supports_correct_2p_xr10_a8,
        supports_cpu_buffers,
        supports_indirect_command_buffers,
        supports_protected_texture_options,
        supports_dispatch_threads_indirect,
        supports_info_indirect,
        supports_display_port_connection_type,
        supports_texture_descriptor2,
        supports_multiple_displays,
    ],
}

const _: () = assert!(size_of::<RawFeatures>() == 60);
const _: () = assert!(std::mem::offset_of!(RawFeatures, max_fifo_count) == 20);
const _: () = assert!(std::mem::offset_of!(RawFeatures, supports_multiple_displays) == 59);

impl GuestFeatures {
    /// The 60 bytes of the struct as the ABI lays it out (the handshake's `features:` by value,
    /// the shared copy's words). The three padding bytes (17-19) are zero. The `u32`s are native,
    /// little-endian on arm64.
    pub fn to_bytes(&self) -> [u8; 60] {
        zerocopy::transmute!(RawFeatures::of(self))
    }

    /// The struct from its 60 bytes. A `BOOL` byte other than 0 or 1 is read as `!= 0` (`bool`
    /// cannot hold it); the host's own writers only store 0 and 1. The padding is ignored.
    pub fn from_bytes(bytes: &[u8; 60]) -> GuestFeatures {
        let raw: RawFeatures = zerocopy::transmute!(*bytes);
        raw.features()
    }
}

/// The device's live `APVFeatures`, which it hands out by reference: tasks, FIFOs and displays read
/// it at each use, so a later binary version write is seen at once. Stored as 15 relaxed atomic
/// words, so a read racing a write may tear between words; no lock is added. Name is ours: the
/// live, shared features.
#[derive(Debug, Default)]
pub struct SharedFeatures {
    words: [AtomicU32; 15],
}

/// A handle on the device's live features.
pub type FeaturesRef = Arc<SharedFeatures>;

impl SharedFeatures {
    /// Zeroed features (a new device's start zeroed).
    pub fn new() -> SharedFeatures {
        SharedFeatures::default()
    }

    /// A snapshot, read word by word.
    pub fn load(&self) -> GuestFeatures {
        let words: [u32; 15] =
            std::array::from_fn(|i| self.words.get(i).map_or(0, |w| w.load(Ordering::Relaxed)));
        let bytes: [u8; 60] = zerocopy::transmute!(words);
        GuestFeatures::from_bytes(&bytes)
    }

    /// Stores every word, in order.
    pub fn store(&self, f: &GuestFeatures) {
        let words: [u32; 15] = zerocopy::transmute!(f.to_bytes());
        for (slot, w) in self.words.iter().zip(words) {
            slot.store(w, Ordering::Relaxed);
        }
    }
}

/// `APVObjectEntry`: packed, 12 bytes; `type:8, length:24` in the first word, then a 64-bit
/// offset at +4.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct ObjectEntry {
    /// `type : 8` (bits 0-7) and `length : 24` (bits 8-31).
    pub type_and_length: u32,
    /// `offset`: the object's serialized record's virtual offset.
    pub offset: u64,
}

const _: () = assert!(size_of::<ObjectEntry>() == 12);

impl ObjectEntry {
    /// `type : 8`: the low byte of the first word (`tst w8, #0xff`).
    pub fn object_type(&self) -> u8 {
        self.type_and_length.low_u8()
    }

    /// `length : 24`: the first word's upper 24 bits (`lsr x3, x8, #8`).
    pub fn length(&self) -> u32 {
        self.type_and_length.wrapping_shr(8)
    }
}

/// `APVObjectPlacement`: a union of `field0` and `{resourceIndex, resourceID}`; 16 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ObjectPlacement {
    /// `field0` / `resource.resourceIndex`: a buffer's or heap's pinned GPU address, or the
    /// resource index another object is created at (`field0`, as the metadata names it: two
    /// roles).
    pub field0: u64,
    /// `resource.resourceID`: the `gpuResourceID` the object gets.
    pub resource_id: u64,
}

const _: () = assert!(size_of::<ObjectPlacement>() == 16);

/// `APVCmdExecIndirectResource`: the 24-byte exec resource record (opcode 55).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the metadata's field names; `fieldNN` where it names none
pub struct CmdExecIndirectResource {
    pub reference: u32,
    pub invalidate_host: u8,
    pub invalidate_guest: u8,
    pub synchronize: u8,
    pub field7: u8,
    pub field8: u64,
    pub field10: u64,
}

const _: () = assert!(size_of::<CmdExecIndirectResource>() == 24);

/// `APVCmdExecIndirectResource3`: the 8-byte record (opcodes 67, 69); `flags` bits 0-2 are
/// `invalidateHost`, `invalidateGuest`, `synchronize`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdExecIndirectResource3 {
    /// `reference`.
    pub reference: u32,
    /// `flags.field0` (the `bits` view is the accessors below).
    pub flags: u32,
}

const _: () = assert!(size_of::<CmdExecIndirectResource3>() == 8);

impl CmdExecIndirectResource3 {
    /// `flags.bits.invalidateHost` (bit 0).
    pub fn invalidate_host(&self) -> bool {
        self.flags & 1u32.wrapping_shl(0) != 0
    }
    /// `flags.bits.invalidateGuest` (bit 1).
    pub fn invalidate_guest(&self) -> bool {
        self.flags & 1u32.wrapping_shl(1) != 0
    }
    /// `flags.bits.synchronize` (bit 2).
    pub fn synchronize(&self) -> bool {
        self.flags & 1u32.wrapping_shl(2) != 0
    }
}

/// `APVCmdExecIndirectCmdBuffer`: the task range a cursor reads.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdExecIndirectCmdBuffer {
    /// `offset`.
    pub offset: u64,
    /// `length`.
    pub length: u64,
}

const _: () = assert!(size_of::<CmdExecIndirectCmdBuffer>() == 16);

/// `APVTaskRoot2`: packed, 40 bytes, at the start of the task root page.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
#[allow(missing_docs)] // the metadata's field names
pub struct TaskRoot2 {
    pub page_table_root: u32,
    pub page_table_depth: u32,
    pub object_list_offset: u64,
    pub object_list_length: u32,
    pub task_length: u64,
    pub placement_list_offset: u64,
    pub placement_list_length: u32,
}

const _: () = assert!(size_of::<TaskRoot2>() == 40);

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GuestFeatures {
        GuestFeatures {
            valid: true,
            object_tables: true,
            buffer_from_io_surface: true,
            max_fifo_count: 0x0102_0304,
            s8_byte_count_for_d32s8: 0x0a0b_0c0d,
            supports_shared_textures: true,
            supports_multiple_displays: true,
            ..GuestFeatures::default()
        }
    }

    /// `GuestFeatures::to_bytes` lays the fields at the ABI's offsets
    #[test]
    fn to_bytes_is_the_abi_layout() {
        let b = sample().to_bytes();
        assert_eq!(b[0], 1); // valid
        assert_eq!(b[1], 0); // unsupported
        assert_eq!(b[2], 1); // objectTables
        assert_eq!(b[16], 1); // bufferFromIOSurface
        assert_eq!(&b[17..20], &[0, 0, 0]); // padding
        assert_eq!(&b[20..24], &[4, 3, 2, 1]); // maxFIFOCount
        assert_eq!(&b[24..28], &[0x0d, 0x0c, 0x0b, 0x0a]); // s8ByteCountForD32S8
        assert_eq!(b[28], 1); // supportsSharedTextures
        assert_eq!(b[59], 1); // supportsMultipleDisplays
        assert_eq!(b.iter().filter(|x| **x == 1).count(), 6);
    }

    /// Every field has its own byte (a field list out of order would move one)
    #[test]
    fn every_bool_has_its_byte() {
        for i in (0..17).chain(28..60) {
            let mut b = [0u8; 60];
            b[i] = 1;
            let f = GuestFeatures::from_bytes(&b);
            assert_eq!(f.to_bytes(), b, "byte {i}");
            assert_ne!(f, GuestFeatures::default());
        }
        let mut b = [0u8; 60];
        b[0] = 1;
        assert!(GuestFeatures::from_bytes(&b).valid);
        b = [0u8; 60];
        b[59] = 1;
        assert!(GuestFeatures::from_bytes(&b).supports_multiple_displays);
        b = [0u8; 60];
        b[50] = 1; // 28 + 22: supportsDisplaySetGuestICCProfile
        assert!(GuestFeatures::from_bytes(&b).supports_display_set_guest_icc_profile);
    }

    /// `from_bytes` round-trips, reads a nonzero `BOOL` byte as YES, ignores padding
    #[test]
    fn from_bytes_round_trip_and_bool_bytes() {
        let f = sample();
        assert_eq!(GuestFeatures::from_bytes(&f.to_bytes()), f);
        let mut b = f.to_bytes();
        b[1] = 0x80;
        b[18] = 0xff;
        let g = GuestFeatures::from_bytes(&b);
        assert!(g.unsupported);
        assert_eq!(g.to_bytes()[1], 1);
        assert_eq!(g.to_bytes()[18], 0);
        assert_eq!(g.max_fifo_count, 0x0102_0304);
    }

    /// `SharedFeatures::new` is zeroed; `store` then `load` gives the struct back
    #[test]
    fn shared_features_store_load() {
        let s = SharedFeatures::new();
        assert_eq!(s.load(), GuestFeatures::default());
        s.store(&sample());
        assert_eq!(s.load(), sample());
        // The words are the struct's 60 bytes: word 5 is maxFIFOCount (offset 20).
        assert_eq!(s.words[5].load(Ordering::Relaxed), 0x0102_0304);
        assert_eq!(s.words[0].load(Ordering::Relaxed), 0x0001_0001);
        s.store(&GuestFeatures::default());
        assert_eq!(s.load(), GuestFeatures::default());
    }

    /// `ObjectEntry::object_type` and `length` split the first word 8/24; offset at +4
    #[test]
    fn object_entry_fields() {
        let bytes: [u8; 12] = [0x0c, 0x34, 0x12, 0xff, 8, 7, 6, 5, 4, 3, 2, 1];
        let e = ObjectEntry::read_from_bytes(&bytes).unwrap();
        assert_eq!(e.object_type(), 0x0c);
        assert_eq!(e.length(), 0xff_1234);
        assert_eq!({ e.offset }, 0x0102_0304_0506_0708);
        let z = ObjectEntry::default();
        assert_eq!((z.object_type(), z.length()), (0, 0));
    }

    /// `CmdExecIndirectResource3`'s bits 0-2
    #[test]
    fn exec_resource3_bits() {
        let r = |flags| CmdExecIndirectResource3 {
            reference: 1,
            flags,
        };
        let get = |x: CmdExecIndirectResource3| {
            (x.invalidate_host(), x.invalidate_guest(), x.synchronize())
        };
        assert_eq!(get(r(0)), (false, false, false));
        assert_eq!(get(r(1)), (true, false, false));
        assert_eq!(get(r(2)), (false, true, false));
        assert_eq!(get(r(4)), (false, false, true));
        assert_eq!(get(r(0xffff_fff8)), (false, false, false));
        assert_eq!(get(r(7)), (true, true, true));
    }

    /// The packed guest records read from their bytes (APVTaskRoot2's 64-bit fields at 0x14 and
    /// 0x1c)
    #[test]
    fn guest_records_from_bytes() {
        let mut b = [0u8; 40];
        b[0x14..0x1c].copy_from_slice(&0x1111_u64.to_le_bytes());
        b[0x1c..0x24].copy_from_slice(&0x2222_u64.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&0x33_u32.to_le_bytes());
        let t = TaskRoot2::read_from_bytes(&b).unwrap();
        assert_eq!(
            ({ t.task_length }, { t.placement_list_offset }),
            (0x1111, 0x2222)
        );
        assert_eq!({ t.placement_list_length }, 0x33);
        let p = ObjectPlacement::read_from_bytes(&[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        assert_eq!((p.field0, p.resource_id), (1, 2));
        let c = CmdExecIndirectResource::read_from_bytes(&[
            9, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        .unwrap();
        assert_eq!(
            (
                c.reference,
                c.invalidate_host,
                c.invalidate_guest,
                c.synchronize
            ),
            (9, 1, 0, 1)
        );
        assert_eq!(APV_MAX_FIFOS, 64);
    }

    /// On 16 KiB guest pages every guest-page constant derived from `APV_PAGE_SHIFT` keeps the
    /// value its 16 KiB literal had
    #[cfg(not(apv_guest_4k))]
    #[test]
    fn guest_page_constants_keep_their_16k_values_on_a_16k_guest() {
        use crate::{device::root, display, fifo::ring, memory::page_table};
        assert_eq!(APV_PAGE_SHIFT, 14);
        assert_eq!(APV_PAGE_SIZE, 0x4000);
        assert_eq!(APV_PAGE_MASK, 0x3fff);
        // The range tracker's rounding.
        assert_eq!(APV_PAGE_SIZE - 2, 0x3ffe);
        assert_eq!(page_table::INDEX_BITS, 12);
        assert_eq!(page_table::ENTRIES, 4096);
        assert_eq!(ring::PAGE_LIST_ENTRIES, 4096);
        assert_eq!(display::efi::PAGE_MASK, 0x3fff);
        assert_eq!(root::PHYSICAL_READ_PAGE, 0x4000);
        assert_eq!(display::page::PAGE_LENGTH, 0x4000);
    }

    /// With 4 KiB guest pages (an x86-64 guest) a page table is 1024 entries of 4 KiB page
    /// numbers, and the display state page's mapping is one of them
    #[cfg(apv_guest_4k)]
    #[test]
    fn guest_page_constants_are_4k_on_a_4k_guest() {
        use crate::{device::root, display, fifo::ring, memory::page_table};
        assert_eq!(APV_PAGE_SHIFT, 12);
        assert_eq!(APV_PAGE_SIZE, 0x1000);
        assert_eq!(APV_PAGE_MASK, 0xfff);
        assert_eq!(APV_PAGE_SIZE - 2, 0xffe);
        assert_eq!(page_table::INDEX_BITS, 10);
        assert_eq!(page_table::ENTRIES, 1024);
        assert_eq!(ring::PAGE_LIST_ENTRIES, 1024);
        assert_eq!(display::efi::PAGE_MASK, 0xfff);
        assert_eq!(root::PHYSICAL_READ_PAGE, 0x1000);
        assert_eq!(display::page::PAGE_LENGTH, 0x1000);
    }
}
