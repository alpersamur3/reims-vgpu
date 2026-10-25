//! Descriptors: the host's description of the device ([`IoSurfaceHostDeviceDescriptor`]), the
//! guest's description of each surface in its two forms, the first form's upgrade to the second,
//! and the IOSurface properties a second-form descriptor makes.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::records::APV_PAGE_SIZE;
use crate::host::{PhysicalMemoryRange, RaiseInterrupt};
use crate::iosurface::table::SurfaceSlot;
use crate::memory::descriptor::MemoryMapDescriptor;
use crate::num::{ToUsize, Truncate};
use crate::platform::Platform;
use crate::platform::os::surface::{SurfaceComponentSpec, SurfacePlaneSpec, SurfaceSpec};

/// The MMIO window a descriptor names by default: one 16 KiB page.
pub const DEFAULT_MMIO_LENGTH: u64 = 16384;

/// What the host gives the device when it makes it. `Default` is the SDK's initial value
/// (`mmio_length` 16384), not an all-zero one. Dropping it releases the interrupt block and the
/// memory map descriptor.
#[derive(Clone)]
pub struct IoSurfaceHostDeviceDescriptor {
    /// The MMIO window's length.
    pub mmio_length: u64,
    /// The block the device raises its interrupt through.
    pub raise_interrupt: Option<RaiseInterrupt>,
    /// The guest's memory; without one the device is refused.
    pub memory_map_descriptor: Option<MemoryMapDescriptor>,
}

impl IoSurfaceHostDeviceDescriptor {
    /// `mmio_length` 16384, no interrupt block, no memory map.
    pub fn new() -> IoSurfaceHostDeviceDescriptor {
        IoSurfaceHostDeviceDescriptor {
            mmio_length: DEFAULT_MMIO_LENGTH,
            raise_interrupt: None,
            memory_map_descriptor: None,
        }
    }
}

impl Default for IoSurfaceHostDeviceDescriptor {
    /// [`IoSurfaceHostDeviceDescriptor::new`].
    fn default() -> Self {
        IoSurfaceHostDeviceDescriptor::new()
    }
}

impl std::fmt::Debug for IoSurfaceHostDeviceDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoSurfaceHostDeviceDescriptor")
            .field("mmio_length", &self.mmio_length)
            .field("raise_interrupt", &self.raise_interrupt.is_some())
            .field(
                "memory_map_descriptor",
                &self.memory_map_descriptor.is_some(),
            )
            .finish()
    }
}

/// The most planes a surface descriptor may name (it has room for 7).
const MAX_PLANES: u8 = 7;
/// The most components a plane may name (its arrays have 4).
const MAX_COMPONENTS: u8 = 4;
/// `kIOSurfaceClientDirection`'s value for a read-only mapping.
const CLIENT_DIRECTION_READ_ONLY: u32 = 2;

/// A descriptor's first word, and an entry of the descriptor table or of a page table:
/// `valid : 1`, then `isTable : 1` (a page-table entry) or `mapped : 1` (a descriptor), then
/// `pageNumber : 30` (the page, or a descriptor's page-table root).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(transparent)]
pub struct IoSurfaceDescriptorPageTableEntry(pub u32);

impl IoSurfaceDescriptorPageTableEntry {
    /// A descriptor's `mapped` bit, in its first word.
    pub(super) const MAPPED: u32 = 2;

    /// `valid` (bit 0).
    pub(super) const fn valid(self) -> bool {
        self.0 & 1 != 0
    }

    /// `isTable` of a page-table entry (bit 1).
    pub(super) const fn is_table(self) -> bool {
        self.0 & 2 != 0
    }

    /// The guest physical address of the page the entry names: its page number shifted by the
    /// guest page shift, in 64 bits.
    pub(super) fn address(self, page_shift: u32) -> u64 {
        u64::from(self.0.wrapping_shr(2)).wrapping_shl(page_shift)
    }
}

/// Which form the guest's descriptors take: bit 0 of the enable register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Form {
    /// [`IoSurfaceDeviceDescriptor`], upgraded when read.
    First,
    /// [`IoSurfaceDeviceDescriptor2`].
    Second,
}

impl Form {
    /// The form the enable register's value selects.
    pub(super) const fn of(enable: u64) -> Form {
        if enable & 1 != 0 {
            Form::Second
        } else {
            Form::First
        }
    }

    /// One descriptor's size in bytes.
    pub(super) const fn size(self) -> NonZeroU64 {
        match self {
            Form::First => const { NonZeroU64::new(0x200).unwrap() },
            Form::Second => const { NonZeroU64::new(0x400).unwrap() },
        }
    }
}

/// A plane of a first-form descriptor (0x40 bytes). `word14` is `elementWidth : 8, width : 24`,
/// `word18` is `elementHeight : 8, height : 24`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the guest ABI's names (docs/testing.md 6)
pub struct IoSurfaceDevicePlaneDescriptor {
    pub field_00: u32,
    pub field_04: u32,
    pub offset: u32,
    pub base: u32,
    pub size: u32,
    pub word14: u32,
    pub word18: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
    pub field_22: u16,
    pub compression_footprint: u8,
    pub address_format: u8,
    pub compression_type: u8,
    pub component_count: u8,
    pub extended_pixels_on_left: u8,
    pub extended_pixels_on_top: u8,
    pub extended_pixels_on_right: u8,
    pub extended_pixels_on_bottom: u8,
    pub component_bit_depths: [u8; 4],
    pub component_bit_offsets: [u8; 4],
    pub component_names: [u8; 4],
    pub component_types: [u8; 4],
    pub component_ranges: [u8; 4],
}

const _: () = assert!(size_of::<IoSurfaceDevicePlaneDescriptor>() == 0x40);

/// The base of a first-form descriptor (0x40 bytes). `word0` is an
/// [`IoSurfaceDescriptorPageTableEntry`] (`valid`, `mapped`, the page-table root); `word14` and
/// `word18` are as in the plane; `bits27` is `readOnly : 1, field_27_1 : 7`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the guest ABI's names (docs/testing.md 6)
pub struct IoSurfaceDeviceBaseDescriptor {
    pub word0: u32,
    pub pixel_format: u32,
    pub offset: u32,
    pub field_0c: u32,
    pub alloc_size: u32,
    pub word14: u32,
    pub word18: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
    pub field_22: u16,
    pub plane_count: u8,
    pub address_format: u8,
    pub compression_type: u8,
    pub bits27: u8,
    pub extended_pixels_on_left: u8,
    pub extended_pixels_on_top: u8,
    pub extended_pixels_on_right: u8,
    pub extended_pixels_on_bottom: u8,
    pub field_2c: u32,
    pub field_30: [u8; 16],
}

const _: () = assert!(size_of::<IoSurfaceDeviceBaseDescriptor>() == 0x40);

/// A first-form descriptor (0x200 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct IoSurfaceDeviceDescriptor {
    /// The surface.
    pub base: IoSurfaceDeviceBaseDescriptor,
    /// Its planes.
    pub planes: [IoSurfaceDevicePlaneDescriptor; 7],
}

const _: () = assert!(size_of::<IoSurfaceDeviceDescriptor>() == 0x200);

/// A plane of a second-form descriptor (0x80 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the guest ABI's names (docs/testing.md 6)
pub struct IoSurfaceDevicePlaneDescriptor2 {
    pub field_00: [u32; 2],
    pub offset: u64,
    pub base: u64,
    pub size: u64,
    pub width: u64,
    pub height: u64,
    pub bytes_per_row: u64,
    pub bytes_per_element: u16,
    pub element_width: u8,
    pub element_height: u8,
    pub compression_footprint: u8,
    pub address_format: u8,
    pub compression_type: u8,
    pub component_count: u8,
    pub extended_pixels_on_left: u32,
    pub extended_pixels_on_top: u32,
    pub extended_pixels_on_right: u32,
    pub extended_pixels_on_bottom: u32,
    pub component_bit_depths: [u8; 4],
    pub component_bit_offsets: [u8; 4],
    pub component_names: [u8; 4],
    pub component_types: [u8; 4],
    pub component_ranges: [u8; 4],
    pub field_64: [u32; 7],
}

const _: () = assert!(size_of::<IoSurfaceDevicePlaneDescriptor2>() == 0x80);

/// The base of a second-form descriptor (0x80 bytes). `word0` is as in the first form; `bits41`
/// is `readOnly : 1, hostOnly : 1, field_41_2 : 6`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the guest ABI's names (docs/testing.md 6)
pub struct IoSurfaceDeviceBaseDescriptor2 {
    pub word0: u32,
    pub pixel_format: u32,
    pub offset: u64,
    pub field_10: u64,
    pub alloc_size: u64,
    pub width: u64,
    pub height: u64,
    pub bytes_per_row: u64,
    pub bytes_per_element: u16,
    pub field_3a: u16,
    pub element_width: u8,
    pub element_height: u8,
    pub plane_count: u8,
    pub address_format: u8,
    pub compression_type: u8,
    pub bits41: u8,
    pub field_42: [u8; 2],
    pub extended_pixels_on_left: u32,
    pub extended_pixels_on_top: u32,
    pub extended_pixels_on_right: u32,
    pub extended_pixels_on_bottom: u32,
    pub field_54: u32,
    pub protection_options: u64,
    pub field_60: [u32; 8],
}

const _: () = assert!(size_of::<IoSurfaceDeviceBaseDescriptor2>() == 0x80);

impl IoSurfaceDeviceBaseDescriptor2 {
    /// `readOnly`: the IOSurface is made read-only for its clients.
    pub(super) const fn read_only(&self) -> bool {
        self.bits41 & 1 != 0
    }

    /// `hostOnly`: the surface has no guest pages; the host allocates it.
    pub(super) const fn host_only(&self) -> bool {
        self.bits41 & 2 != 0
    }
}

/// A second-form descriptor (0x400 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct IoSurfaceDeviceDescriptor2 {
    /// The surface.
    pub base: IoSurfaceDeviceBaseDescriptor2,
    /// Its planes.
    pub planes: [IoSurfaceDevicePlaneDescriptor2; 7],
}

const _: () = assert!(size_of::<IoSurfaceDeviceDescriptor2>() == 0x400);

/// An element width or height byte, decoded: a byte with bit 7 set is a power of two,
/// `1 << (v & 0x7f)`, else the byte itself. The shift is a 32-bit `lsl` of the whole byte, so its
/// count is taken mod 32.
pub fn element_dimension(value: u8) -> u32 {
    let v = u32::from(value);
    if v & 0x80 != 0 {
        crate::num::bit32(v)
    } else {
        v
    }
}

/// Widens a first-form descriptor into `descriptor`, field by field. `hostOnly`, `field_3a`,
/// `field_41_2`, `field_42`, `protection_options` and `field_60` are zeroed; each plane's
/// `field_64` is left as it was (never read).
pub fn upgrade_descriptor(
    descriptor: &mut IoSurfaceDeviceDescriptor2,
    original: &IoSurfaceDeviceDescriptor,
) {
    let from = &original.base;
    let to = &mut descriptor.base;
    // `valid`, `mapped` and the root: the whole word.
    to.word0 = from.word0;
    to.pixel_format = from.pixel_format;
    to.offset = u64::from(from.offset);
    to.field_10 = u64::from(from.field_0c);
    to.alloc_size = u64::from(from.alloc_size);
    to.element_width = from.word14.low_u8();
    to.element_height = from.word18.low_u8();
    to.bytes_per_row = u64::from(from.bytes_per_row);
    to.bytes_per_element = from.bytes_per_element;
    to.field_3a = 0;
    to.width = u64::from(from.word14.wrapping_shr(8));
    to.height = u64::from(from.word18.wrapping_shr(8));
    to.plane_count = from.plane_count;
    to.address_format = from.address_format;
    to.compression_type = from.compression_type;
    // `readOnly`; `hostOnly` and `field_41_2` zero.
    to.bits41 = from.bits27 & 1;
    to.field_42 = [0, 0];
    to.extended_pixels_on_left = u32::from(from.extended_pixels_on_left);
    to.extended_pixels_on_top = u32::from(from.extended_pixels_on_top);
    to.extended_pixels_on_right = u32::from(from.extended_pixels_on_right);
    to.extended_pixels_on_bottom = u32::from(from.extended_pixels_on_bottom);
    to.field_54 = from.field_2c;
    to.protection_options = 0;
    to.field_60 = [0; 8];
    for (plane, from) in descriptor.planes.iter_mut().zip(&original.planes) {
        plane.field_00 = [from.field_00, from.field_04];
        plane.offset = u64::from(from.offset);
        plane.base = u64::from(from.base);
        plane.size = u64::from(from.size);
        plane.element_width = from.word14.low_u8();
        plane.element_height = from.word18.low_u8();
        plane.bytes_per_row = u64::from(from.bytes_per_row);
        plane.bytes_per_element = from.bytes_per_element;
        plane.width = u64::from(from.word14.wrapping_shr(8));
        plane.height = u64::from(from.word18.wrapping_shr(8));
        plane.compression_footprint = from.compression_footprint;
        plane.address_format = from.address_format;
        plane.compression_type = from.compression_type;
        plane.component_count = from.component_count;
        plane.extended_pixels_on_left = u32::from(from.extended_pixels_on_left);
        plane.extended_pixels_on_top = u32::from(from.extended_pixels_on_top);
        plane.extended_pixels_on_right = u32::from(from.extended_pixels_on_right);
        plane.extended_pixels_on_bottom = u32::from(from.extended_pixels_on_bottom);
        plane.component_bit_depths = from.component_bit_depths;
        plane.component_bit_offsets = from.component_bit_offsets;
        plane.component_names = from.component_names;
        plane.component_types = from.component_types;
        plane.component_ranges = from.component_ranges;
    }
}

/// One plane's IOSurface properties; `None` for more than four components. The numbers' widths
/// are the properties' (`u64` or `u32`), as the IOSurface dictionary holds them.
pub(super) fn plane_spec(plane: &IoSurfaceDevicePlaneDescriptor2) -> Option<SurfacePlaneSpec> {
    let mut spec = SurfacePlaneSpec {
        base: plane.base,
        offset: plane.offset,
        width: plane.width,
        height: plane.height,
        size: plane.size,
        bytes_per_row: plane.bytes_per_row,
        bytes_per_element: u64::from(plane.bytes_per_element),
        element_width: element_dimension(plane.element_width),
        element_height: element_dimension(plane.element_height),
        compression_footprint: u32::from(plane.compression_footprint),
        address_format: u32::from(plane.address_format),
        compression_type: u32::from(plane.compression_type),
        extended_pixels_on_left: plane.extended_pixels_on_left,
        extended_pixels_on_top: plane.extended_pixels_on_top,
        extended_pixels_on_right: plane.extended_pixels_on_right,
        extended_pixels_on_bottom: plane.extended_pixels_on_bottom,
        components: Vec::new(),
    };
    let count = plane.component_count;
    if count > MAX_COMPONENTS {
        log::warn!("a plane has {count} components, more than {MAX_COMPONENTS}");
        return None;
    }
    // No components: no component arrays.
    spec.components = plane
        .component_bit_depths
        .iter()
        .zip(&plane.component_bit_offsets)
        .zip(&plane.component_names)
        .zip(&plane.component_types)
        .zip(&plane.component_ranges)
        .take(usize::from(count))
        .map(
            |((((&bit_depth, &bit_offset), &name), &type_), &range)| SurfaceComponentSpec {
                bit_depth,
                bit_offset,
                name,
                type_,
                range,
            },
        )
        .collect();
    Some(spec)
}

/// The first `plane_count` planes' properties (the caller checked the count is at most 7);
/// `None` when a plane's are refused.
pub(super) fn plane_specs(
    descriptor: &IoSurfaceDeviceDescriptor2,
) -> Option<Vec<SurfacePlaneSpec>> {
    let count = usize::from(descriptor.base.plane_count);
    let mut planes = Vec::with_capacity(count);
    for plane in descriptor.planes.iter().take(count) {
        planes.push(plane_spec(plane)?);
    }
    Some(planes)
}

/// The IOSurface properties of `descriptor`, and the slot's base offset. A guest-backed surface
/// lists each of its guest pages as a client address range of one guest page; a host-only one
/// lists none. The surface-level compression type goes under the plane key, as the device has
/// always sent it.
pub(super) fn surface_spec<P: Platform>(
    descriptor: &IoSurfaceDeviceDescriptor2,
    slot: &mut SurfaceSlot<P>,
) -> Option<SurfaceSpec> {
    let base = &descriptor.base;
    let planes = if base.plane_count == 0 {
        None
    } else if base.plane_count > MAX_PLANES {
        log::warn!(
            "a surface has {} planes, more than {MAX_PLANES}",
            base.plane_count
        );
        return None;
    } else {
        // The plane's line says why.
        let Some(planes) = plane_specs(descriptor) else {
            log::debug!("a surface's {} planes were refused", base.plane_count);
            return None;
        };
        Some(planes)
    };
    let mut spec = SurfaceSpec {
        planes,
        pixel_format: base.pixel_format,
        offset: base.offset,
        alloc_size: base.alloc_size,
        width: base.width,
        height: base.height,
        bytes_per_element: u64::from(base.bytes_per_element),
        bytes_per_row: base.bytes_per_row,
        element_width: element_dimension(base.element_width),
        element_height: element_dimension(base.element_height),
        address_format: u32::from(base.address_format),
        compression_type: u32::from(base.compression_type),
        extended_pixels_on_left: base.extended_pixels_on_left,
        extended_pixels_on_top: base.extended_pixels_on_top,
        extended_pixels_on_right: base.extended_pixels_on_right,
        extended_pixels_on_bottom: base.extended_pixels_on_bottom,
        protection_options: base.protection_options,
        client_direction: None,
        client_address_ranges: None,
    };
    if slot.read_only {
        log::debug!("the surface is read-only for its clients");
        spec.client_direction = Some(CLIENT_DIRECTION_READ_ONLY);
    }
    slot.base_offset = base.offset;
    if slot.host_only {
        log::debug!("the surface is host-only");
        return Some(spec);
    }
    let page = APV_PAGE_SIZE;
    spec.client_address_ranges = Some(
        slot.pages
            .iter()
            .take(slot.page_count.to_usize())
            .map(|&address| PhysicalMemoryRange {
                physical_address: address,
                physical_length: page,
            })
            .collect(),
    );
    Some(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn init_defaults() {
        for d in [
            IoSurfaceHostDeviceDescriptor::new(),
            IoSurfaceHostDeviceDescriptor::default(),
        ] {
            assert_eq!(d.mmio_length, 16384);
            assert!(d.raise_interrupt.is_none());
            assert!(d.memory_map_descriptor.is_none());
        }
    }

    /// Fields; the last drop releases the block
    #[test]
    fn accessors_and_drop() {
        let mut d = IoSurfaceHostDeviceDescriptor::new();
        d.mmio_length = 0;
        let ri: RaiseInterrupt = Arc::new(|_| {});
        let weak = Arc::downgrade(&ri);
        d.raise_interrupt = Some(ri);
        assert_eq!(d.mmio_length, 0);
        let s = format!("{d:?}");
        assert!(s.contains("raise_interrupt: true") && s.contains("memory_map_descriptor: false"));
        drop(d);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn plane_specs_decode_dimensions_and_cap_components() {
        let mut d = IoSurfaceDeviceDescriptor2::default();
        d.base.plane_count = 2;
        d.planes[0].element_width = 0x83;
        d.planes[0].element_height = 0xa0;
        d.planes[0].component_count = 3;
        d.planes[0].component_ranges = [1, 2, 3, 4];
        d.planes[0].bytes_per_element = 0xffff;
        d.planes[1].address_format = 9;
        let planes = plane_specs(&d).unwrap();
        assert_eq!(planes.len(), 2);
        assert_eq!((planes[0].element_width, planes[0].element_height), (8, 1));
        assert_eq!(planes[0].bytes_per_element, 0xffff);
        assert_eq!(
            planes[0]
                .components
                .iter()
                .map(|c| c.range)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(planes[1].components.is_empty());
        assert_eq!(planes[1].address_format, 9);
        d.planes[1].component_count = 0xff;
        assert!(plane_specs(&d).is_none());
        assert!(plane_spec(&d.planes[1]).is_none());
    }

    #[test]
    fn element_dimension_shifts_mod_32() {
        assert_eq!(element_dimension(0), 0);
        assert_eq!(element_dimension(0x7f), 0x7f);
        assert_eq!(element_dimension(0x80), 1);
        assert_eq!(element_dimension(0x84), 16);
        assert_eq!(element_dimension(0x9f), 1 << 31);
        // `lsl w` takes the count mod 32: 0xa0 & 0x7f = 32 shifts by 0.
        assert_eq!(element_dimension(0xa0), 1);
        assert_eq!(element_dimension(0xff), 1 << 31);
    }

    #[test]
    fn upgrade_widens_every_field() {
        let mut v1 = IoSurfaceDeviceDescriptor::default();
        v1.base.word0 = 0xffff_fffb;
        v1.base.pixel_format = 1;
        v1.base.offset = 2;
        v1.base.field_0c = 3;
        v1.base.alloc_size = 4;
        v1.base.word14 = 0x1234_5605;
        v1.base.word18 = 0xffff_ff06;
        v1.base.bytes_per_row = 7;
        v1.base.bytes_per_element = 8;
        v1.base.field_22 = 9;
        v1.base.plane_count = 10;
        v1.base.address_format = 11;
        v1.base.compression_type = 12;
        v1.base.bits27 = 0xfe;
        v1.base.extended_pixels_on_left = 13;
        v1.base.extended_pixels_on_top = 14;
        v1.base.extended_pixels_on_right = 15;
        v1.base.extended_pixels_on_bottom = 16;
        v1.base.field_2c = 17;
        v1.base.field_30 = [0xee; 16];
        for (i, p) in v1.planes.iter_mut().enumerate() {
            let i = i as u32;
            p.field_00 = 100 + i;
            p.field_04 = 200 + i;
            p.offset = 1 + i;
            p.base = 2 + i;
            p.size = 3 + i;
            p.word14 = ((0x10 + i) << 8) | 4;
            p.word18 = ((0x20 + i) << 8) | 5;
            p.bytes_per_row = 6;
            p.bytes_per_element = 7;
            p.field_22 = 0xffff;
            p.compression_footprint = 8;
            p.address_format = 9;
            p.compression_type = 10;
            p.component_count = 11;
            p.extended_pixels_on_left = 0xf1;
            p.extended_pixels_on_top = 0xf2;
            p.extended_pixels_on_right = 0xf3;
            p.extended_pixels_on_bottom = 0xf4;
            p.component_bit_depths = [1, 2, 3, 4];
            p.component_bit_offsets = [5, 6, 7, 8];
            p.component_names = [9, 10, 11, 12];
            p.component_types = [13, 14, 15, 16];
            p.component_ranges = [17, 18, 19, 20];
        }
        let mut d = IoSurfaceDeviceDescriptor2::default();
        d.base.bits41 = 0xff;
        d.base.field_3a = 0xffff;
        d.base.protection_options = 9;
        d.base.field_60 = [1; 8];
        d.base.field_42 = [1, 1];
        d.planes[2].field_64 = [0x55; 7];
        upgrade_descriptor(&mut d, &v1);
        let b = &d.base;
        assert_eq!(b.word0, 0xffff_fffb);
        assert_eq!(
            (b.pixel_format, b.offset, b.field_10, b.alloc_size),
            (1, 2, 3, 4)
        );
        assert_eq!((b.element_width, b.width), (5, 0x12_3456));
        assert_eq!((b.element_height, b.height), (6, 0xff_ffff));
        assert_eq!(
            (b.bytes_per_row, b.bytes_per_element, b.field_3a),
            (7, 8, 0)
        );
        assert_eq!(
            (b.plane_count, b.address_format, b.compression_type),
            (10, 11, 12)
        );
        assert_eq!((b.bits41, b.field_42), (0, [0, 0]));
        assert_eq!(
            [
                b.extended_pixels_on_left,
                b.extended_pixels_on_top,
                b.extended_pixels_on_right,
                b.extended_pixels_on_bottom
            ],
            [13, 14, 15, 16]
        );
        assert_eq!(
            (b.field_54, b.protection_options, b.field_60),
            (17, 0, [0; 8])
        );
        v1.base.bits27 = 1;
        upgrade_descriptor(&mut d, &v1);
        assert_eq!(d.base.bits41, 1);
        for (i, p) in d.planes.iter().enumerate() {
            let i = i as u64;
            assert_eq!(p.field_00, [100 + i as u32, 200 + i as u32]);
            assert_eq!((p.offset, p.base, p.size), (1 + i, 2 + i, 3 + i));
            assert_eq!((p.width, p.height), (0x10 + i, 0x20 + i));
            assert_eq!((p.element_width, p.element_height), (4, 5));
            assert_eq!((p.bytes_per_row, p.bytes_per_element), (6, 7));
            assert_eq!(
                (
                    p.compression_footprint,
                    p.address_format,
                    p.compression_type,
                    p.component_count
                ),
                (8, 9, 10, 11)
            );
            assert_eq!(
                [
                    p.extended_pixels_on_left,
                    p.extended_pixels_on_top,
                    p.extended_pixels_on_right,
                    p.extended_pixels_on_bottom
                ],
                [0xf1, 0xf2, 0xf3, 0xf4]
            );
            assert_eq!(p.component_ranges, [17, 18, 19, 20]);
            assert_eq!(p.component_types, [13, 14, 15, 16]);
        }
        // `field_64` is not written.
        assert_eq!(d.planes[2].field_64, [0x55; 7]);
    }
}
