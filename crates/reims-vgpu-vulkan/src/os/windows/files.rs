//! The process's backing table on Windows: which section, from which offset, each byte of the
//! memory the library may make a view of shows, so that a view is made by mapping sections
//! (`MapViewOfFile3`) and never by remapping addresses, which Win32 has no call for (Mach's
//! `mach_vm_remap`); and which of those bytes are views this VM mapped, as Win32 unmaps a view
//! only whole (`UnmapViewOfFile2`).
//!
//! Two kinds of entry, as on Linux (`os/linux/files.rs`):
//! - **Registrations** of memory the library does not map itself: the guest's RAM, the host's
//!   views of the pagefile sections it names in its memory map (the `guest_memory` lease of the
//!   memory map's device). One registered again over the same section and offset is counted,
//!   and goes with the last. The C API duplicates the host's handles for each device it makes,
//!   so devices over the same RAM register it each under its own: a byte resolves through the
//!   first registration that holds it, and every one shows the same section pages.
//! - **Runs** the library mapped: every view [`super::vm::WindowsVm`] made, and every range of
//!   [`super::vm::allocate_aliasable`]. A run shows `end - first` bytes of its section from its
//!   offset, with the protection last set on it (`PAGE_*`). Runs never overlap: a new run
//!   replaces what of another it covers, and an unmap or an overwrite drops what of the runs it
//!   covers.
//!
//! A byte in a run is found there first; a registration is looked at only for bytes no run
//! holds. Because a view of a view resolves to the section, never to the first view's address,
//! the second view keeps showing the pages after the first goes, as Mach's aliases do.
//!
//! **Views.** Beside the runs, the table keeps each view the VM mapped (`MapViewOfFile3` over a
//! placeholder): its bytes, its section and the section offset of its first byte. Every byte of
//! a view lies in a run of the same section at the same offset (a run may be cut inside a view
//! where its protection was set apart), which is what lets the VM map the rest of a view again
//! after it unmapped the view whole to take part of it ([`super::vm`]). Registrations are no
//! views: the host mapped them, and the host unmaps them.
//!
//! The table is a leaf lock: nothing is called under it. The VM's calls that map and unmap
//! (`remap_shared_overwrite`, `allocate_fixed_overwrite`, `deallocate`) are serialized by
//! [`mutation`], a lock held across their Win32 calls, so two threads never carve or map the
//! same placeholders at once; no call of the library's runs under it.

use std::collections::BTreeMap;
use std::os::windows::io::OwnedHandle;
use std::sync::Arc;

use reims_vgpu_core::sync::{Mutex, MutexGuard};
use windows_sys::Win32::System::Memory::{PAGE_PROTECTION_FLAGS, PAGE_READWRITE};

use crate::os::GuestRamFile;

/// The section behind a range: an open handle (`CreateFileMappingW`'s, or the host's duplicated),
/// and the section offset of the range's first byte.
#[derive(Clone, Debug)]
pub(crate) struct Backing {
    /// The section, held while any entry or view names it.
    pub(crate) section: Arc<OwnedHandle>,
    /// The section offset of the range's first byte.
    pub(crate) offset: u64,
}

impl Backing {
    /// Whether both name the same section (the same handle object, not the same kernel object:
    /// a duplicate of the host's handle is another backing, which resolves to the same pages).
    fn same_section(&self, other: &Backing) -> bool {
        Arc::ptr_eq(&self.section, &other.section)
    }
}

/// A run of the library's own mapping: its end, its section and the protection set on it.
#[derive(Clone, Debug)]
struct Run {
    end: u64,
    backing: Backing,
    prot: PAGE_PROTECTION_FLAGS,
}

/// A registration: `first..end` shows `backing`, read and write, `count` times registered.
#[derive(Debug)]
struct Registered {
    first: u64,
    end: u64,
    backing: Backing,
    count: usize,
}

/// A view the VM mapped: `first..end` is one `MapViewOfFile3` of `backing` (module docs).
#[derive(Clone, Debug)]
struct MappedView {
    end: u64,
    backing: Backing,
}

/// One piece of a range as the table resolves it: `length` bytes, `at` bytes into the range,
/// showing `backing`, with protection `prot`.
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub(crate) at: u64,
    pub(crate) length: u64,
    pub(crate) backing: Backing,
    pub(crate) prot: PAGE_PROTECTION_FLAGS,
}

/// A view as the table lists it: its first byte, its length and its section from its offset.
#[derive(Clone, Debug)]
pub(crate) struct View {
    pub(crate) first: u64,
    pub(crate) length: u64,
    pub(crate) backing: Backing,
}

/// The table (module docs).
#[derive(Debug, Default)]
struct Table {
    registered: Vec<Registered>,
    runs: BTreeMap<u64, Run>,
    views: BTreeMap<u64, MappedView>,
}

/// The process's table.
static TABLE: Mutex<Table> = Mutex::new(Table {
    registered: Vec::new(),
    runs: BTreeMap::new(),
    views: BTreeMap::new(),
});

/// The lock the VM's mapping calls hold across their Win32 calls (module docs).
static MUTATION: Mutex<()> = Mutex::new(());

/// Takes the VM's mutation lock (module docs): held while a call carves placeholders and maps or
/// unmaps views, so no other thread does so meanwhile.
pub(crate) fn mutation() -> MutexGuard<'static, ()> {
    MUTATION.lock()
}

impl Table {
    /// The run holding byte `at`: its first byte and itself.
    fn run_at(&self, at: u64) -> Option<(u64, &Run)> {
        (self.runs.range(..=at).next_back())
            .filter(|(_, run)| at < run.end)
            .map(|(&first, run)| (first, run))
    }

    /// The first run starting after `at`, before `end`: its first byte.
    fn next_run(&self, at: u64, end: u64) -> Option<u64> {
        self.runs.range(at..end).next().map(|(&first, _)| first)
    }

    /// The registration holding byte `at`.
    fn registered_at(&self, at: u64) -> Option<&Registered> {
        self.registered.iter().find(|r| r.first <= at && at < r.end)
    }

    /// Drops what of the runs lies in `first..end`, keeping the rest of each.
    fn forget(&mut self, first: u64, end: u64) {
        let meeting: Vec<(u64, Run)> = (self.runs.range(..end).rev())
            .take_while(|(_, run)| run.end > first)
            .map(|(&at, run)| (at, run.clone()))
            .collect();
        for (at, run) in meeting {
            self.runs.remove(&at);
            if at < first {
                let mut head = run.clone();
                head.end = first;
                self.runs.insert(at, head);
            }
            if end < run.end {
                let mut tail = run.clone();
                tail.backing.offset = run.backing.offset + (end - at);
                self.runs.insert(end, tail);
            }
        }
    }

    /// `first..end` shows `backing` with protection `prot` from now on (`record`).
    fn record(&mut self, first: u64, end: u64, backing: Backing, prot: PAGE_PROTECTION_FLAGS) {
        self.forget(first, end);
        if first < end {
            self.runs.insert(first, Run { end, backing, prot });
        }
    }

    /// Sets the protection of what of the runs lies in `first..end` (`protect`).
    fn protect(&mut self, first: u64, end: u64, prot: PAGE_PROTECTION_FLAGS) {
        let meeting: Vec<(u64, Run)> = (self.runs.range(..end).rev())
            .take_while(|(_, run)| run.end > first)
            .map(|(&at, run)| (at, run.clone()))
            .collect();
        for (at, run) in meeting {
            let (from, to) = (at.max(first), run.end.min(end));
            self.forget(from, to);
            let mut changed = run.clone();
            changed.end = to;
            changed.backing.offset = run.backing.offset + (from - at);
            changed.prot = prot;
            self.runs.insert(from, changed);
        }
    }

    /// The view holding byte `at`.
    fn view_at(&self, at: u64) -> Option<View> {
        (self.views.range(..=at).next_back())
            .filter(|(_, view)| at < view.end)
            .map(|(&first, view)| View {
                first,
                length: view.end - first,
                backing: view.backing.clone(),
            })
    }
}

/// `length` bytes from `first` show `backing` from now on, with protection `prot`: what of
/// another run they cover goes.
pub(crate) fn record(first: u64, length: u64, backing: Backing, prot: PAGE_PROTECTION_FLAGS) {
    TABLE
        .lock()
        .record(first, first.saturating_add(length), backing, prot);
}

/// `length` bytes from `first` show no section from now on.
pub(crate) fn forget(first: u64, length: u64) {
    TABLE.lock().forget(first, first.saturating_add(length));
}

/// The pieces of `length` bytes from `first`, each the bytes of one run or registration in
/// order, those of a run cut where it ends; `None` where a byte has no section behind it.
pub(crate) fn pieces(first: u64, length: u64) -> Option<Vec<Piece>> {
    let end = first.checked_add(length)?;
    let table = TABLE.lock();
    let mut pieces = Vec::new();
    let mut at = first;
    while at < end {
        if let Some((run_first, run)) = table.run_at(at) {
            let to = run.end.min(end);
            let mut backing = run.backing.clone();
            backing.offset += at - run_first;
            pieces.push(Piece {
                at: at - first,
                length: to - at,
                backing,
                prot: run.prot,
            });
            at = to;
            continue;
        }
        let registered = table.registered_at(at)?;
        let to = (registered.end.min(end)).min(table.next_run(at, end).unwrap_or(end));
        let mut backing = registered.backing.clone();
        backing.offset += at - registered.first;
        pieces.push(Piece {
            at: at - first,
            length: to - at,
            backing,
            prot: PAGE_READWRITE,
        });
        at = to;
    }
    Some(pieces)
}

/// The pieces of `length` bytes from `first` that lie in the table's runs alone, in order, with
/// their gaps: `Some` for a run's bytes, `None` for bytes in no run, each gap one piece. Which
/// bytes a put-back maps again from their sections and which it fills with fresh pages
/// ([`super::vm`]).
pub(crate) fn run_pieces(first: u64, length: u64) -> Vec<(u64, u64, Option<Piece>)> {
    let end = first.saturating_add(length);
    let table = TABLE.lock();
    let mut out = Vec::new();
    let mut at = first;
    while at < end {
        if let Some((run_first, run)) = table.run_at(at) {
            let to = run.end.min(end);
            let mut backing = run.backing.clone();
            backing.offset += at - run_first;
            out.push((
                at,
                to - at,
                Some(Piece {
                    at: at - first,
                    length: to - at,
                    backing,
                    prot: run.prot,
                }),
            ));
            at = to;
        } else {
            let to = table.next_run(at, end).unwrap_or(end);
            out.push((at, to - at, None));
            at = to;
        }
    }
    out
}

/// Sets the protection of what of the runs lies in `length` bytes from `first` to `prot`,
/// cutting the runs it meets at its ends.
pub(crate) fn protect(first: u64, length: u64, prot: PAGE_PROTECTION_FLAGS) {
    TABLE
        .lock()
        .protect(first, first.saturating_add(length), prot);
}

/// Whether a byte of `length` bytes from `first` lies in a registration.
pub(crate) fn meets_registration(first: u64, length: u64) -> bool {
    let end = first.saturating_add(length);
    let table = TABLE.lock();
    table
        .registered
        .iter()
        .any(|r| r.first < end && first < r.end)
}

/// Whether `length` bytes from `first` lie wholly in one registration: the host's own committed
/// read-write pages (the guest's RAM), which the VM knows without asking Win32. `VirtualQuery`
/// of a page in a view of a large section is slow: it walks the view's page entries from the
/// page to where they stop matching, about all of the guest's 4 GiB at each call.
pub(crate) fn within_registration(first: u64, length: u64) -> bool {
    let end = first.saturating_add(length);
    let table = TABLE.lock();
    table
        .registered
        .iter()
        .any(|r| r.first <= first && end <= r.end)
}

/// `length` bytes from `first` are one view of `backing` from now on (module docs).
pub(crate) fn record_view(first: u64, length: u64, backing: Backing) {
    let end = first.saturating_add(length);
    let mut table = TABLE.lock();
    // A view never lies over another: the VM unmaps what it replaces first.
    let meeting: Vec<u64> = (table.views.range(..end).rev())
        .take_while(|(_, view)| view.end > first)
        .map(|(&at, _)| at)
        .collect();
    for at in meeting {
        table.views.remove(&at);
    }
    if first < end {
        table.views.insert(first, MappedView { end, backing });
    }
}

/// The view that starts at `first` is unmapped from now on: itself, as it was listed.
pub(crate) fn forget_view(first: u64) -> Option<View> {
    let mut table = TABLE.lock();
    let view = table.views.remove(&first)?;
    Some(View {
        first,
        length: view.end - first,
        backing: view.backing,
    })
}

/// The view holding the byte `at`, as listed.
pub(crate) fn view_at(at: u64) -> Option<View> {
    TABLE.lock().view_at(at)
}

/// How many views the table lists in `first..end`, for tests.
#[cfg(test)]
pub(crate) fn views_between(first: u64, end: u64) -> usize {
    TABLE.lock().views.range(first..end).count()
}

/// A registration of `length` bytes from `first` as showing `backing` (module docs), until it
/// drops.
#[derive(Debug)]
pub(crate) struct Registration {
    first: u64,
    end: u64,
    backing: Backing,
}

impl Registration {
    /// Registers `length` bytes from `first` as showing `backing`: counted again where the same
    /// bytes are registered over the same section and offset.
    pub(crate) fn new(first: u64, length: u64, backing: Backing) -> Registration {
        let end = first.saturating_add(length);
        let mut table = TABLE.lock();
        match table
            .registered
            .iter_mut()
            .find(|r| r.is(first, end, &backing))
        {
            Some(r) => r.count += 1,
            None => table.registered.push(Registered {
                first,
                end,
                backing: backing.clone(),
                count: 1,
            }),
        }
        Registration {
            first,
            end,
            backing,
        }
    }
}

/// The guest's RAM registered while a device made over it lives: the `guest_memory` lease its
/// device holds, made of the host's memory map, and dropped after core's teardown. Views of the
/// guest's pages are made of these sections.
#[derive(Debug)]
pub struct GuestMemoryLease {
    _registrations: Vec<Registration>,
}

/// Registers each range of the guest's RAM the host names a section behind: counted where the
/// same range was registered already over the same handle (module docs). A range without one is
/// not registered, and a view of its pages is refused as a failed remap is.
pub fn lease_guest_memory(ranges: impl IntoIterator<Item = GuestRamFile>) -> GuestMemoryLease {
    let registrations: Vec<Registration> = ranges
        .into_iter()
        .map(|r| {
            let backing = Backing {
                section: r.fd,
                offset: r.offset,
            };
            Registration::new(r.host, r.length, backing)
        })
        .collect();
    GuestMemoryLease {
        _registrations: registrations,
    }
}

impl Registered {
    /// Whether this registers `first..end` over `backing`.
    fn is(&self, first: u64, end: u64, backing: &Backing) -> bool {
        (self.first, self.end, self.backing.offset) == (first, end, backing.offset)
            && self.backing.same_section(backing)
    }
}

impl Drop for Registration {
    /// Counts the registration down, and drops it at the last.
    fn drop(&mut self) {
        let mut table = TABLE.lock();
        if let Some(at) =
            (table.registered.iter()).position(|r| r.is(self.first, self.end, &self.backing))
        {
            let r = &mut table.registered[at];
            r.count -= 1;
            if r.count == 0 {
                table.registered.remove(at);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY};

    use super::super::vm::section;
    use super::*;

    /// A section for the tests' backings.
    fn file() -> Arc<OwnedHandle> {
        section(0x10_0000).unwrap()
    }

    /// The pieces' (offset in the range, length, section offset).
    fn shape(pieces: &[Piece]) -> Vec<(u64, u64, u64)> {
        pieces
            .iter()
            .map(|p| (p.at, p.length, p.backing.offset))
            .collect()
    }

    /// Whether every piece of `length` bytes from `first` reads, by the table alone.
    fn readable(first: u64, length: u64) -> Option<bool> {
        let pieces = pieces(first, length)?;
        Some(pieces.iter().all(|p| super::super::vm::reads(p.prot)))
    }

    // The table is the process's: each test uses address ranges of its own, far apart, which
    // nothing maps.

    /// A range resolves through the runs over it, cut where they end, then the registrations;
    /// a byte of neither leaves it unresolved
    #[test]
    fn a_range_resolves_through_runs_then_registrations() {
        let f = file();
        let base = 0x7a00_0000_0000;
        let backing = |offset| Backing {
            section: Arc::clone(&f),
            offset,
        };
        let held = Registration::new(base, 0x10_000, backing(0x100_000));
        record(base + 0x20_000, 0x2000, backing(0x5000), PAGE_READONLY);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x2000).unwrap()),
            [(0, 0x2000, 0x101_000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x2000, 0x5000)]
        );
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
        assert!(pieces(base + 0x1f_000, 0x2000).is_none());
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(true));
        assert_eq!(readable(base, 0x10_000), Some(true));
        protect(base + 0x21_000, 0x1000, PAGE_NOACCESS);
        assert_eq!(readable(base + 0x20_000, 0x1000), Some(true));
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(false));
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x1000, 0x5000), (0x1000, 0x1000, 0x6000)]
        );
        assert_eq!(readable(base + 0x30_000, 1), None);
        drop(held);
        assert!(pieces(base, 0x1000).is_none());
        forget(base + 0x20_000, 0x2000);
    }

    /// A run replaces what of another it covers, and an unmap drops what it covers, keeping
    /// each run's other bytes at their section offsets; the run pieces list the gaps
    #[test]
    fn runs_never_overlap() {
        let f = file();
        let base = 0x7b00_0000_0000;
        let backing = |offset| Backing {
            section: Arc::clone(&f),
            offset,
        };
        record(base, 0x4000, backing(0x10_000), PAGE_READWRITE);
        record(base + 0x1000, 0x1000, backing(0x90_000), PAGE_READONLY);
        assert_eq!(
            shape(&pieces(base, 0x4000).unwrap()),
            [
                (0, 0x1000, 0x10_000),
                (0x1000, 0x1000, 0x90_000),
                (0x2000, 0x2000, 0x12_000)
            ]
        );
        forget(base + 0x2000, 0x1000);
        assert!(pieces(base, 0x4000).is_none());
        assert_eq!(
            shape(&pieces(base + 0x3000, 0x1000).unwrap()),
            [(0, 0x1000, 0x13_000)]
        );
        let gaps: Vec<(u64, u64, bool)> = run_pieces(base, 0x4000)
            .into_iter()
            .map(|(at, length, piece)| (at - base, length, piece.is_some()))
            .collect();
        assert_eq!(
            gaps,
            [
                (0, 0x1000, true),
                (0x1000, 0x1000, true),
                (0x2000, 0x1000, false),
                (0x3000, 0x1000, true)
            ]
        );
        forget(base, 0x4000);
        assert!(pieces(base, 1).is_none());
        assert_eq!(run_pieces(base, 0x1000).len(), 1);
    }

    /// A lease registers each range of the guest's RAM at its section offset, until it drops
    #[test]
    fn a_lease_registers_the_guests_ram_while_held() {
        let f = file();
        let base = 0x7d00_0000_0000;
        let ram = |host, length, offset| GuestRamFile {
            host,
            length,
            fd: Arc::clone(&f),
            offset,
        };
        let lease =
            lease_guest_memory([ram(base, 0x4000, 0), ram(base + 0x10_000, 0x2000, 0x4000)]);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x1000).unwrap()),
            [(0, 0x1000, 0x1000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x11_000, 0x1000).unwrap()),
            [(0, 0x1000, 0x5000)]
        );
        assert!(pieces(base + 0x4000, 0x1000).is_none());
        assert!(meets_registration(base + 0x3000, 0x2000));
        assert!(!meets_registration(base + 0x4000, 0x1000));
        drop(lease);
        assert!(pieces(base, 0x1000).is_none());
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
    }

    /// The same bytes registered again over the same section and offset are counted: they stay
    /// until the last registration drops
    #[test]
    fn registrations_are_counted() {
        let f = file();
        let base = 0x7c00_0000_0000;
        let backing = || Backing {
            section: Arc::clone(&f),
            offset: 0,
        };
        let first = Registration::new(base, 0x1000, backing());
        let second = Registration::new(base, 0x1000, backing());
        drop(first);
        assert!(pieces(base, 0x1000).is_some());
        drop(second);
        assert!(pieces(base, 0x1000).is_none());
    }

    /// A view is listed from its record to its forget, found by any byte of it, and a view
    /// recorded over another replaces it
    #[test]
    fn views_are_listed_while_mapped() {
        let f = file();
        let base = 0x7e00_0000_0000;
        let backing = |offset| Backing {
            section: Arc::clone(&f),
            offset,
        };
        record_view(base, 0x3000, backing(0x8000));
        let v = view_at(base + 0x2fff).unwrap();
        assert_eq!((v.first, v.length, v.backing.offset), (base, 0x3000, 0x8000));
        assert!(view_at(base + 0x3000).is_none());
        assert_eq!(views_between(base, base + 0x10_000), 1);
        record_view(base + 0x1000, 0x1000, backing(0));
        assert!(view_at(base).is_none(), "the first view went");
        assert_eq!(views_between(base, base + 0x10_000), 1);
        let gone = forget_view(base + 0x1000).unwrap();
        assert_eq!((gone.first, gone.length), (base + 0x1000, 0x1000));
        assert!(forget_view(base + 0x1000).is_none());
        assert_eq!(views_between(base, base + 0x10_000), 0);
    }
}
