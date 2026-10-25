//! The IOSurface device's test rig: a device over 16 pages of fake guest RAM, with helpers that
//! write the guest's ring, table, descriptors and page tables.

#![allow(dead_code)] // each test module uses its part

use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use zerocopy::IntoBytes;

use crate::host::PhysicalMemoryRange;
use crate::iosurface::descriptor::{
    IoSurfaceDeviceDescriptor, IoSurfaceDeviceDescriptor2, IoSurfaceHostDeviceDescriptor,
    element_dimension,
};
use crate::iosurface::device::{HostDeviceCore, HostDeviceHandle, IoSurfaceHostDevice};
use crate::iosurface::regs::{
    REG_DESC_ENABLE, REG_DESCRIPTOR_BASE, REG_RING_BASE, REG_RING_SIZE, REG_RING_WR_COUNTER,
};
use crate::iosurface::ring::IoReturn;
use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
use crate::platform::os::surface::SurfaceSpec;
use crate::platform::{Handle, ObjId, Platform};
use crate::testkit::calllog::Arg;
use crate::testkit::guest_ram::GuestRam;
use crate::testkit::logs::Capture;
use crate::testkit::platform::FakePlatform;

pub(super) type Dev = IoSurfaceHostDevice<FakePlatform>;

/// The guest page, which the mapper's tables count (the fake VM's host page may be smaller).
pub(super) const PAGE: u64 = crate::guest::records::APV_PAGE_SIZE;
/// The descriptor pages, from `DPAGE` on: 16 KiB, 16 second-form or 32 first-form descriptors,
/// one page on 16 KiB guest pages and four on 4 KiB ones.
pub(super) const DESCRIPTOR_PAGES: u64 = 0x4000 / PAGE;
/// Guest RAM at 1 MiB: 16 pages on 16 KiB guest pages, and ten pages from `SURF` on on every
/// guest.
pub(super) const B: u64 = 0x10_0000;
pub(super) const RAM_PAGES: u64 = 15 + DESCRIPTOR_PAGES;
pub(super) const RING: u64 = B;
pub(super) const TABLE: u64 = B + PAGE;
pub(super) const DPAGE: u64 = B + 2 * PAGE;
pub(super) const ROOT: u64 = DPAGE + DESCRIPTOR_PAGES * PAGE;
pub(super) const L2A: u64 = ROOT + PAGE;
pub(super) const L2B: u64 = ROOT + 2 * PAGE;
pub(super) const SURF: u64 = ROOT + 3 * PAGE;
/// Outside every range.
pub(super) const HOLE: u64 = 0x4000_0000;

pub(super) struct Rig {
    pub(super) p: Arc<FakePlatform>,
    pub(super) ram: GuestRam,
    pub(super) dev: HostDeviceHandle<FakePlatform>,
    pub(super) irqs: Arc<Mutex<Vec<u32>>>,
}

pub(super) fn map_descriptor(ram: &GuestRam) -> MemoryMapDescriptor {
    let mut d = MemoryMapDescriptor::new();
    for r in ram.ranges() {
        d.add_range(GuestPhysicalRange {
            physical_address: r.physical_address,
            physical_length: r.region.len(),
            host_address: r.host(),
        });
    }
    d
}

pub(super) fn platform_and_ram() -> (Arc<FakePlatform>, GuestRam) {
    let p = Arc::new(FakePlatform::new());
    let ram = GuestRam::with_range(B, RAM_PAGES * PAGE).unwrap();
    p.fake_vm().adopt(&ram);
    (p, ram)
}

pub(super) fn rig_with(interrupt: bool) -> Rig {
    let (p, ram) = platform_and_ram();
    let irqs = Arc::new(Mutex::new(Vec::new()));
    let i2 = irqs.clone();
    let mut d = IoSurfaceHostDeviceDescriptor::new();
    if interrupt {
        d.raise_interrupt = Some(Arc::new(move |v| i2.lock().push(v)));
    }
    d.memory_map_descriptor = Some(map_descriptor(&ram));
    let dev = Dev::new(&p, &d).unwrap();
    Rig { p, ram, dev, irqs }
}

pub(super) fn rig() -> Rig {
    rig_with(true)
}

/// A rig whose interrupt block runs `on_raise` (the host's block).
pub(super) fn rig_raising(on_raise: impl Fn(u32) + Send + Sync + 'static) -> Rig {
    let (p, ram) = platform_and_ram();
    let mut d = IoSurfaceHostDeviceDescriptor::new();
    d.raise_interrupt = Some(Arc::new(on_raise));
    d.memory_map_descriptor = Some(map_descriptor(&ram));
    let dev = Dev::new(&p, &d).unwrap();
    Rig {
        p,
        ram,
        dev,
        irqs: Arc::new(Mutex::new(Vec::new())),
    }
}

/// A page-table entry naming `pa`.
pub(super) fn pte(pa: u64, table: bool) -> u32 {
    (((pa >> crate::guest::records::APV_PAGE_SHIFT) as u32) << 2) | 1 | if table { 2 } else { 0 }
}

/// A valid second-form descriptor of `pages` guest pages whose root table is at `root`.
pub(super) fn v2(pages: u64, root: u64) -> IoSurfaceDeviceDescriptor2 {
    let mut d = IoSurfaceDeviceDescriptor2::default();
    d.base.word0 = pte(root, false);
    d.base.pixel_format = 0x4247_5241;
    d.base.alloc_size = pages * PAGE;
    d.base.width = 64;
    d.base.height = 32;
    d.base.bytes_per_row = 256;
    d.base.bytes_per_element = 4;
    d.base.element_width = 1;
    d.base.element_height = 1;
    d.base.address_format = 3;
    d.base.compression_type = 2;
    d.base.extended_pixels_on_left = 5;
    d.base.extended_pixels_on_top = 6;
    d.base.extended_pixels_on_right = 7;
    d.base.extended_pixels_on_bottom = 8;
    d.base.protection_options = 0x77;
    d
}

pub(super) fn spec_for(
    d: &IoSurfaceDeviceDescriptor2,
    ranges: Option<Vec<PhysicalMemoryRange>>,
) -> SurfaceSpec {
    SurfaceSpec {
        planes: None,
        pixel_format: d.base.pixel_format,
        offset: d.base.offset,
        alloc_size: d.base.alloc_size,
        width: d.base.width,
        height: d.base.height,
        bytes_per_element: u64::from(d.base.bytes_per_element),
        bytes_per_row: d.base.bytes_per_row,
        element_width: element_dimension(d.base.element_width),
        element_height: element_dimension(d.base.element_height),
        address_format: u32::from(d.base.address_format),
        compression_type: u32::from(d.base.compression_type),
        extended_pixels_on_left: d.base.extended_pixels_on_left,
        extended_pixels_on_top: d.base.extended_pixels_on_top,
        extended_pixels_on_right: d.base.extended_pixels_on_right,
        extended_pixels_on_bottom: d.base.extended_pixels_on_bottom,
        protection_options: d.base.protection_options,
        client_direction: None,
        client_address_ranges: ranges,
    }
}

impl Rig {
    pub(super) fn sync(&self) {
        self.p.fake_vm().sync();
    }

    /// Ring of 1024 commands at `RING`, descriptor table at `TABLE` whose entries name the
    /// descriptor pages from `DPAGE` on, in the first (`v2 == false`) or second form.
    pub(super) fn setup(&self, v2: bool) {
        for i in 0..DESCRIPTOR_PAGES {
            self.ram
                .write_u32(TABLE + 4 * i, pte(DPAGE + i * PAGE, false));
        }
        if v2 {
            self.dev.mmio_write(REG_DESC_ENABLE, 1);
        }
        self.dev.mmio_write(REG_RING_BASE, RING);
        self.dev.mmio_write(REG_RING_SIZE, 1024);
        self.dev.mmio_write(REG_DESCRIPTOR_BASE, TABLE);
    }

    /// Root table at `ROOT` naming `pages` pages from `SURF` on.
    pub(super) fn root(&self, pages: u64) {
        for i in 0..pages {
            self.ram
                .write_u32(ROOT + 4 * i, pte(SURF + i * PAGE, false));
        }
    }

    pub(super) fn put_v2(&self, id: u64, d: &IoSurfaceDeviceDescriptor2) {
        assert!(self.ram.write(DPAGE + (id % 16) * 0x400, d.as_bytes()));
        self.sync();
    }

    pub(super) fn put_v1(&self, id: u64, d: &IoSurfaceDeviceDescriptor) {
        assert!(self.ram.write(DPAGE + (id % 32) * 0x200, d.as_bytes()));
        self.sync();
    }

    pub(super) fn word0_v2(&self, id: u64) -> u32 {
        self.sync();
        self.ram.read_u32(DPAGE + (id % 16) * 0x400).unwrap()
    }

    pub(super) fn word0_v1(&self, id: u64) -> u32 {
        self.sync();
        self.ram.read_u32(DPAGE + (id % 32) * 0x200).unwrap()
    }

    pub(super) fn command(&self, slot: u64, command: u32, id: u32) {
        self.ram.write_u32(RING + slot * 16, command);
        self.ram.write_u32(RING + slot * 16 + 4, id);
        self.ram.write_u32(RING + slot * 16 + 8, 0x5555);
    }

    pub(super) fn status(&self, slot: u64) -> i32 {
        self.sync();
        self.ram.read_u32(RING + slot * 16 + 8).unwrap() as i32
    }

    /// Writes the write counter and waits for the ring batch it wakes.
    pub(super) fn doorbell(&self, wr: u64) {
        self.sync();
        self.dev.mmio_write(REG_RING_WR_COUNTER, wr);
        self.dev.hw_queue.sync(|| ());
        self.sync();
    }

    pub(super) fn surface(&self, id: usize) -> Option<ObjId> {
        self.dev.state.lock().slots[id]
            .surface
            .as_ref()
            .map(Handle::id)
    }

    pub(super) fn held(&self, id: usize) -> <FakePlatform as Platform>::Surface {
        self.dev.state.lock().slots[id]
            .surface
            .as_ref()
            .unwrap()
            .clone_ref()
    }

    pub(super) fn created(&self) -> Vec<Arg> {
        self.p
            .call_log()
            .calls_to("IOSurfaceCreate")
            .into_iter()
            .map(|c| c.args[0].clone())
            .collect()
    }

    pub(super) fn host(&self, pa: u64) -> u64 {
        self.ram.host_addr(pa).unwrap().addr() as u64
    }

    /// A map command's work, on the calling thread.
    pub(super) fn map(&self, id: u32) -> IoReturn {
        self.dev.map(&mut self.dev.state.lock(), id)
    }

    /// An unmap command's work, on the calling thread.
    pub(super) fn unmap(&self, id: u32) -> IoReturn {
        self.dev.unmap(&mut self.dev.state.lock(), id)
    }

    /// A test command's work, on the calling thread.
    pub(super) fn test(&self, id: u32) -> IoReturn {
        self.dev.test_pattern(&self.dev.state.lock(), id)
    }

    /// Tears every surface down, on the calling thread.
    pub(super) fn teardown(&self) {
        self.dev.teardown_surfaces(&mut self.dev.state.lock());
    }

    /// A guest-backed second-form surface of `pages` pages at id `id`, mapped.
    pub(super) fn mapped_v2(&self, id: u32, pages: u64) {
        self.setup(true);
        self.root(pages);
        self.put_v2(u64::from(id), &v2(pages, ROOT));
        assert_eq!(self.map(id), IoReturn::Success);
    }
}

/// The lines `logs` took since its last take that the module under test wrote, each as its
/// level and message. `test_module` is the test's `module_path!()`: the module under test is the
/// path above its `tests`.
pub(super) fn lines_from(logs: &Capture, test_module: &str) -> Vec<(log::Level, String)> {
    let target = test_module
        .find("::tests")
        .map_or(test_module, |at| &test_module[..at]);
    logs.take()
        .into_iter()
        .filter(|l| l.1 == target)
        .map(|l| (l.0, l.2))
        .collect()
}

/// A gate the host's block waits on, and a count of its entries.
#[derive(Default)]
pub(super) struct Gate {
    pub(super) entered: Mutex<u32>,
    open: Mutex<bool>,
    cv: parking_lot::Condvar,
}

impl Gate {
    pub(super) fn enter_and_wait(&self) {
        *self.entered.lock() += 1;
        self.cv.notify_all();
        let mut open = self.open.lock();
        while !*open {
            self.cv.wait(&mut open);
        }
    }

    pub(super) fn wait_entered(&self) {
        let mut tries = 0u32;
        while *self.entered.lock() == 0 {
            tries += 1;
            assert!(tries < 10_000, "the handler never raised");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    pub(super) fn open(&self) {
        *self.open.lock() = true;
        self.cv.notify_all();
    }
}

/// Waits until the teardown of the device whose state is `core` has taken the doorbell: it has
/// released the mappings and the interrupt block, and its wait for the queue is next.
pub(super) fn wait_doorbell_taken(core: &Weak<HostDeviceCore<FakePlatform>>) {
    let mut tries = 0u32;
    while core.upgrade().is_some_and(|c| c.doorbell.lock().is_some()) {
        tries += 1;
        assert!(tries < 10_000, "the teardown never took the doorbell");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Waits until `p`'s call log has `sel` since `mark`.
pub(super) fn wait_called(p: &FakePlatform, mark: u64, sel: &str) {
    let mut tries = 0u32;
    while !p.call_log().sels_since(mark).contains(&sel) {
        tries += 1;
        assert!(tries < 10_000, "never called {sel}");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// The `mach_vm_deallocate`s since `mark`.
pub(super) fn deallocations(p: &FakePlatform, mark: u64) -> usize {
    p.call_log()
        .sels_since(mark)
        .iter()
        .filter(|s| **s == "mach_vm_deallocate")
        .count()
}
