//! The page walk: the host address of each guest page of a guest-backed surface, through the
//! page table its descriptor roots. A surface of at most one table page of entries uses the root
//! as its only level; a larger one uses the root's entries as second-level tables, each of up to
//! one page of entries.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;

use crate::guest::GuestPtr;
use crate::iosurface::DeathKind;
use crate::iosurface::descriptor::{
    IoSurfaceDescriptorPageTableEntry as Entry, IoSurfaceDeviceDescriptor2,
};
use crate::iosurface::device::HostDeviceCore;
use crate::iosurface::ring::IoReturn;
use crate::iosurface::table::{SurfaceSlot, TABLE_ENTRY_SIZE};
use crate::num::{self, ToU64, ToUsize};
use crate::platform::Platform;

impl<P: Platform> HostDeviceCore<P> {
    /// Fills `slot` with the `page_count` pages (`page` bytes each, `1 << shift`) of the surface
    /// `descriptor` describes. The root
    /// must be mapped, and the surface at most a full two-level table; a refusal is NoMemory,
    /// except a bad root entry, which is BadArgument.
    pub(super) fn walk(
        &self,
        mapping_id: u32,
        descriptor: &IoSurfaceDeviceDescriptor2,
        page_count: u64,
        (page, shift): (u64, u32),
        slot: &mut SurfaceSlot<P>,
    ) -> Result<(), IoReturn> {
        let root_address = Entry(descriptor.base.word0).address(shift);
        let root = self.virtual_address_for_physical(root_address, page);
        if root.is_null() {
            log::warn!(
                "map: surface {mapping_id}'s page table root at {root_address:#x} is not in guest \
                 memory"
            );
            return Err(IoReturn::NoMemory);
        }
        let root = GuestPtr::new(root);
        let entries_per_page = page / const { NonZeroU64::new(TABLE_ENTRY_SIZE).unwrap() };
        if page_count > entries_per_page.wrapping_mul(entries_per_page) {
            log::warn!(
                "map: surface {mapping_id} has {page_count} pages, more than two levels of page \
                 table hold"
            );
            return Err(IoReturn::NoMemory);
        }
        let table_count = num::div0(
            page_count.wrapping_add(entries_per_page).wrapping_sub(1),
            entries_per_page,
        );
        // Never, after the check above.
        if table_count > entries_per_page {
            log::warn!(
                "map: surface {mapping_id} needs {table_count} page tables, more than the root \
                 holds"
            );
            return Err(IoReturn::NoMemory);
        }
        slot.page_count = page_count;
        slot.pages = vec![0; page_count.to_usize()];
        slot.read_only = descriptor.base.read_only();
        slot.host_only = false;
        if table_count > 1 {
            return self.walk_two_levels(
                mapping_id,
                root,
                table_count,
                page_count,
                entries_per_page,
                &mut slot.pages,
            );
        }
        for i in 0..page_count {
            let entry =
                Entry(root.read_u32(i.wrapping_mul(TABLE_ENTRY_SIZE), DeathKind::PageTableRead));
            // A bad root entry is BadArgument.
            if !entry.valid() || entry.is_table() {
                log::warn!(
                    "map: surface {mapping_id}'s page table entry {i} ({:#x}) does not name a \
                     page",
                    entry.0
                );
                return Err(IoReturn::BadArgument);
            }
            let address = entry.address(shift);
            let host_page = self.virtual_address_for_physical(address, page);
            if host_page.is_null() {
                log::warn!(
                    "map: surface {mapping_id}'s page {i} at {address:#x} is not in guest memory"
                );
                return Err(IoReturn::NoMemory);
            }
            if let Some(p) = slot.pages.get_mut(i.to_usize()) {
                *p = host_page.addr().to_u64();
            }
        }
        Ok(())
    }

    /// The two-level walk: each root entry names a table of up to one page of entries.
    fn walk_two_levels(
        &self,
        mapping_id: u32,
        root: GuestPtr,
        table_count: u64,
        page_count: u64,
        entries_per_page: u64,
        pages: &mut [u64],
    ) -> Result<(), IoReturn> {
        let page = self.page_size();
        let shift = self.page_shift();
        let mut remaining = page_count;
        let mut mapped: u64 = 0;
        for i in 0..table_count {
            let table_entry =
                Entry(root.read_u32(i.wrapping_mul(TABLE_ENTRY_SIZE), DeathKind::PageTableRead));
            // A bad root entry is BadArgument.
            if !table_entry.valid() || !table_entry.is_table() {
                log::warn!(
                    "map: surface {mapping_id}'s root entry {i} ({:#x}) does not name a page table",
                    table_entry.0
                );
                return Err(IoReturn::BadArgument);
            }
            let table_address = table_entry.address(shift);
            let table = self.virtual_address_for_physical(table_address, page);
            if table.is_null() {
                log::warn!(
                    "map: surface {mapping_id}'s page table {i} at {table_address:#x} is not in \
                     guest memory"
                );
                return Err(IoReturn::NoMemory);
            }
            let table = GuestPtr::new(table);
            let count = remaining.min(entries_per_page);
            for j in 0..count {
                let entry = Entry(
                    table.read_u32(j.wrapping_mul(TABLE_ENTRY_SIZE), DeathKind::PageTableRead),
                );
                // A bad second-level entry is NoMemory.
                if !entry.valid() || entry.is_table() {
                    log::warn!(
                        "map: surface {mapping_id}'s page table {i} entry {j} ({:#x}) does not \
                         name a page",
                        entry.0
                    );
                    return Err(IoReturn::NoMemory);
                }
                let address = entry.address(shift);
                let host_page = self.virtual_address_for_physical(address, page);
                if host_page.is_null() {
                    log::warn!(
                        "map: surface {mapping_id}'s page {mapped} at {address:#x} is not in guest \
                         memory"
                    );
                    return Err(IoReturn::NoMemory);
                }
                if let Some(p) = pages.get_mut(mapped.to_usize()) {
                    *p = host_page.addr().to_u64();
                }
                mapped = mapped.wrapping_add(1);
            }
            remaining = remaining.wrapping_sub(count);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::iosurface::testing::*;
    use crate::testkit::calllog::Arg;

    /// The root table must be mapped, and the page count at most a page of entries squared,
    /// 4096 × 4096 on 16 KiB guest pages.
    #[test]
    fn root_and_size_refusals() {
        let r = rig();
        r.setup(true);
        r.put_v2(1, &v2(1, HOLE));
        assert_eq!(r.map(1), IoReturn::NoMemory);
        let mut d = v2(1, ROOT);
        let n = PAGE / 4;
        d.base.alloc_size = n * n * PAGE + 1;
        r.put_v2(1, &d);
        assert_eq!(r.map(1), IoReturn::NoMemory);
        assert!(r.created().is_empty());
    }

    /// One table level: an invalid or table root entry is BadArgument; an unmapped page
    /// NoMemory; the pages are freed.
    #[test]
    fn single_level_refusals() {
        let r = rig();
        r.setup(true);
        r.root(2);
        r.ram.write_u32(ROOT + 4, pte(SURF, true));
        r.put_v2(1, &v2(2, ROOT));
        assert_eq!(r.map(1), IoReturn::BadArgument);
        r.ram.write_u32(ROOT + 4, 0);
        assert_eq!(r.map(1), IoReturn::BadArgument);
        r.ram.write_u32(ROOT + 4, pte(HOLE, false));
        assert_eq!(r.map(1), IoReturn::NoMemory);
        let st = r.dev.state.lock();
        let slot = &st.slots[1];
        assert!(slot.pages.is_empty() && slot.surface.is_none());
    }

    /// Two table levels (one page more than a page of entries, 4097 pages on 16 KiB guest pages):
    /// the root entries must be valid tables (BadArgument), the tables mapped and their entries
    /// valid pages (NoMemory).
    #[test]
    fn two_levels() {
        let r = rig();
        r.setup(true);
        let n = PAGE / 4;
        let d = v2(n + 1, ROOT);
        r.put_v2(1, &d);
        r.ram.write_u32(ROOT, pte(L2A, true));
        r.ram.write_u32(ROOT + 4, pte(L2B, true));
        for i in 0..n {
            r.ram
                .write_u32(L2A + 4 * i, pte(SURF + (i % 2) * PAGE, false));
        }
        r.ram.write_u32(L2B, pte(SURF, false));
        r.ram.write_u32(L2B + 4, 0); // past the count: never read
        assert_eq!(r.map(1), IoReturn::Success);
        {
            let st = r.dev.state.lock();
            let pages = &st.slots[1].pages;
            assert_eq!(pages.len(), (n + 1).to_usize());
            assert_eq!(pages[1], r.host(SURF + PAGE));
            assert_eq!(pages[n.to_usize()], r.host(SURF));
        }
        let Arg::Val(text) = &r.created()[0] else {
            panic!()
        };
        // 67125248 on 16 KiB guest pages.
        assert!(text.contains(&format!("alloc_size: {}", (n + 1) * PAGE)));
        // Refusals, on a fresh id each.
        r.ram.write_u32(ROOT + 4, pte(L2B, false));
        r.put_v2(2, &d);
        assert_eq!(r.map(2), IoReturn::BadArgument);
        r.ram.write_u32(ROOT + 4, pte(HOLE, true));
        assert_eq!(r.map(2), IoReturn::NoMemory);
        r.ram.write_u32(ROOT + 4, pte(L2B, true));
        r.ram.write_u32(L2B, pte(SURF, true));
        assert_eq!(r.map(2), IoReturn::NoMemory);
        r.ram.write_u32(L2B, pte(HOLE, false));
        assert_eq!(r.map(2), IoReturn::NoMemory);
        assert_eq!(r.created().len(), 1);
    }
}
