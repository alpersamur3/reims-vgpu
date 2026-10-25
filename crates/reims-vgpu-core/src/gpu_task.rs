//! The GPU task service: one isolated guest task, run on threads of its own
//! (`tasks::isolation`).
//!
//! [`GpuTask`] is the service the host's remote task talks to over its connection
//! ([`ServiceSide`](crate::tasks::launcher::ServiceSide)). The handshake runs synchronously on
//! the connection's thread; every other message queues a job on the task's serial service
//! queue. Jobs run one at a time in arrival order, each in its own real pool and holder, with no
//! raise handler live and as the task's code (`TaskLife::run`), so a raise no job catches ends
//! the task at the raise, which the host sees as the task's death. The XR10 blit pipelines and
//! the resource-index rebuild are the task's own, made or done once per task.
//!
//! The deaths a guest reaches in the task's own code end the task, not the process: a command
//! buffer the task has no cursor over (the segment walk's `NullSegment`), a read or a reply through
//! a task page outside guest RAM (`memory`), compute info of pipeline reference 0
//! ([`DeathKind::NoComputePipeline`]), a task page table whose root is page 0 (the walk's
//! `WalkZeroTable`), a raise out of a rate map's reset (`exec`), and the backing lookup of a
//! reference texture ([`Unsupported::BackingForId`]). A crash inside Metal ends the process
//! (`tasks::isolation::life`).
//!
//! - [`service`]: the task, its message dispatch, the list, resource and FIFO messages, and
//!   what it provides to its resources;
//! - [`handshake`]: the task's setup;
//! - [`memory`]: task memory through the guest's page table (mappings, copies, cursors,
//!   address ranges, buffers over task pages) and the object lists' mapping;
//! - [`paging`]: the resources an exec pages in before it runs and out after it;
//! - [`exec`]: the execs and infos, their completions, and the compute-info, heap-texture and
//!   rate-map queries;
//! - [`surfaces`]: the IOSurface cache and the shared-texture handles, host round trips;
//! - [`blit`]: the XR10 blit pipelines, made once per task.
//!
//! Ordering contracts (each named test pins it):
//! - **The handshake** holds the task lock throughout, so the set-object-list job it queues
//!   runs after it. Its Metal sends come in a fixed order: the device lookup, then
//!   `allowLibrariesFromOtherPlatforms`, then (after the memory map and the task root) the
//!   command queue, the paging queue and the paging event, then the parts, then the capability
//!   queries. A missing map, a zero length or a task root that fails reply 2; with no device it
//!   replies 2 and carries on to reply 1 over parts built on nil (`handshake::tests::
//!   builds_the_task`, `without_a_device_carries_on`, `task_root_failures_reply_2`).
//! - **Catch sites** are fixed: each logs and replies 2; any other raise in a job ends the task
//!   (`exec::tests::ensure_failure_replies_2_per_message`, `reset_rate_map_zero_is_refused`).
//! - **The exec** takes the shared task lock, pages its resources in on the paging queue, waits
//!   on the paging event or the FIFO's, decodes, adds its completion, stores the FIFO's next
//!   value before it encodes the scheduled signal, commits, then pages synchronized resources
//!   out behind the exec (`exec::tests::exec_commits_then_completes`,
//!   `exec_waits_for_the_fifo`).
//! - **The info** waits up to 5 s for the previous exec to be scheduled, decodes without a
//!   commit, and signals on the CPU (`exec::tests::info_signals_without_a_commit`).
//! - **Paging's cleanup** ends the open encoders, bumps the paging counter, signals and
//!   commits before a raise from the encoding goes on; a raise before it goes on at once
//!   (`paging::tests::paging_out_order_and_an_early_raise`).
//! - **Surfaces**: a retain holds the cache's lock across its host round trip; the release
//!   that ends a surface's count tells the host under that lock (`surfaces::tests::
//!   surfaces_are_cached_and_counted`).
//! - **Guest stores**: a compute-info entry is one paired store, never torn inside a 16-byte
//!   granule (`exec::tests::compute_info_pairs_are_never_torn_in_a_granule`); the task length
//!   goes into the task root (`handshake::tests::builds_the_task`).

mod blit;
mod exec;
mod handshake;
mod memory;
mod paging;
mod service;
mod surfaces;

pub use service::GpuTask;

use std::fmt;

use crate::raise::RaiseId;
use crate::raise::RaiseResult;
use crate::raise::death::DeathSite;
use crate::raise::{OnFault, Raise};

/// This area's raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// A range of task memory past the task's end.
    OutOfRange,
    /// Task memory mapped before the handshake set up the page table.
    NoPageTable,
    /// A mapping at an offset that is not page-aligned.
    UnalignedOffset,
    /// A mapping whose length is not page-aligned.
    UnalignedLength,
    /// A mapping of no bytes.
    ZeroLength,
    /// No mapping could be made for an address a command writes through.
    NoMapping,
    /// The mapping a command writes through is read-only.
    ReadOnlyMapping,
    /// A guest read longer than the largest task (32 GiB).
    BufferLength,
    /// The object lists could not be mapped.
    ObjectListsUnmapped,
    /// No mapping could be made for a buffer over task pages.
    NoBufferMapping,
    /// Metal made no buffer over task pages.
    NoBuffer,
    /// A reply's length outside 16 to a guest page less one (0x3fff, 0xfff on 4 KiB guest pages)
    /// bytes.
    ReplyLength,
}

impl RaiseKind {
    /// The raise's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            RaiseKind::OutOfRange => "gpu_task.range",
            RaiseKind::NoPageTable => "gpu_task.page_table.none",
            RaiseKind::UnalignedOffset => "gpu_task.mapping.unaligned_offset",
            RaiseKind::UnalignedLength => "gpu_task.mapping.unaligned_length",
            RaiseKind::ZeroLength => "gpu_task.mapping.zero_length",
            RaiseKind::NoMapping => "gpu_task.mapping.none",
            RaiseKind::ReadOnlyMapping => "gpu_task.mapping.read_only",
            RaiseKind::BufferLength => "gpu_task.buffer.length",
            RaiseKind::ObjectListsUnmapped => "gpu_task.object_lists.unmapped",
            RaiseKind::NoBufferMapping => "gpu_task.buffer.no_mapping",
            RaiseKind::NoBuffer => "gpu_task.buffer.none",
            RaiseKind::ReplyLength => "gpu_task.reply.length",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::GpuTask(k)
    }
}

/// An operation of the shared task core that the service does not provide: the task ends where a
/// resource reaches one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unsupported {
    /// Paging a backing in through the task.
    PrepareBacking,
    /// Paging a backing out through the task.
    SynchronizeBacking,
    /// The backing a reference texture names (a ref texture ends the task).
    BackingForId,
}

/// This area's deaths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A message that needs the task's parts before a handshake built them (the host never
    /// sends one).
    NotSetUp,
    /// The task root is read or written through a NULL address.
    NullTaskRoot,
    /// The task root page is 0 (the handshake refuses it before).
    ZeroTaskRoot,
    /// A reply is written through a NULL address.
    NullReply,
    /// A request's records are shorter than its count says.
    ShortRecords,
    /// A copy's chunk would pass its buffer.
    CopyChunk,
    /// A cursor's walk would pass its length.
    CursorLength,
    /// The shader library the blit pipelines come from is missing.
    NoBlitLibrary,
    /// The XR10 blit-in pipeline could not be made.
    NoBlitInPipeline,
    /// The XR10 blit-out pipeline could not be made.
    NoBlitOutPipeline,
    /// An exec that synchronizes resources has no FIFO event to wait on.
    SyncWithoutFifoEvent,
    /// An exec that paged resources in has no paging event to wait on.
    SyncWithoutPagingEvent,
    /// A surface released that the cache does not hold.
    UnknownSurface,
    /// A surface released that is not the one the cache holds.
    WrongSurface,
    /// Compute info asked of pipeline reference 0.
    NoComputePipeline,
    /// An operation the service does not provide.
    Unsupported(Unsupported),
}

impl DeathKind {
    /// The death's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::NotSetUp => "gpu_task.not_set_up",
            DeathKind::NullTaskRoot => "gpu_task.root.null",
            DeathKind::ZeroTaskRoot => "gpu_task.root.zero",
            DeathKind::NullReply => "gpu_task.reply.null",
            DeathKind::ShortRecords => "gpu_task.records.short",
            DeathKind::CopyChunk => "gpu_task.copy.chunk",
            DeathKind::CursorLength => "gpu_task.cursor.length",
            DeathKind::NoBlitLibrary => "gpu_task.blit.library",
            DeathKind::NoBlitInPipeline => "gpu_task.blit.in_pipeline",
            DeathKind::NoBlitOutPipeline => "gpu_task.blit.out_pipeline",
            DeathKind::SyncWithoutFifoEvent => "gpu_task.sync.fifo_event",
            DeathKind::SyncWithoutPagingEvent => "gpu_task.sync.paging_event",
            DeathKind::UnknownSurface => "gpu_task.surface.unknown",
            DeathKind::WrongSurface => "gpu_task.surface.wrong",
            DeathKind::NoComputePipeline => "gpu_task.compute_info.pipeline",
            DeathKind::Unsupported(Unsupported::PrepareBacking) => {
                "gpu_task.unsupported.prepare_backing"
            }
            DeathKind::Unsupported(Unsupported::SynchronizeBacking) => {
                "gpu_task.unsupported.synchronize_backing"
            }
            DeathKind::Unsupported(Unsupported::BackingForId) => {
                "gpu_task.unsupported.backing_for_id"
            }
        }
    }

    /// The site a death of this kind records: the kind of fault, and the id.
    pub const fn site(self) -> DeathSite {
        use crate::raise::death::DeathKind as Fault;
        let kind = match self {
            DeathKind::NotSetUp | DeathKind::NullTaskRoot | DeathKind::NullReply => {
                Fault::NullDeref
            }
            DeathKind::ShortRecords => Fault::OutOfBounds,
            DeathKind::UnknownSurface | DeathKind::WrongSurface | DeathKind::Unsupported(_) => {
                Fault::Abort
            }
            DeathKind::ZeroTaskRoot
            | DeathKind::CopyChunk
            | DeathKind::CursorLength
            | DeathKind::NoBlitLibrary
            | DeathKind::NoBlitInPipeline
            | DeathKind::NoBlitOutPipeline
            | DeathKind::SyncWithoutFifoEvent
            | DeathKind::SyncWithoutPagingEvent
            | DeathKind::NoComputePipeline => Fault::Assert,
        };
        DeathSite {
            kind,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for crate::raise::DeathId {
    fn from(k: DeathKind) -> crate::raise::DeathId {
        crate::raise::DeathId::GpuTask(k)
    }
}

/// Raises `kind` with `detail` in a function reached both ways ([`OnFault::throw`]): with
/// [`OnFault::Die`] the task dies here (its thread unwinds to the task's run), or the process
/// outside a task's code; with [`OnFault::Return`], the raise is the caller's.
#[track_caller]
fn raise_on<T>(on: OnFault, kind: RaiseKind, detail: impl fmt::Display) -> RaiseResult<T> {
    on.throw(Raise::new(kind).with_detail(detail))
}

/// Raises `kind` with `detail` at a site no path reaches without a handler: `Err`, the caller's
/// to catch.
#[track_caller]
fn raise_caught<T>(kind: RaiseKind, detail: impl fmt::Display) -> RaiseResult<T> {
    Err(Raise::new(kind).with_detail(detail))
}

/// Raises `kind` with `detail` as [`raise_caught`] does, carrying `v` outward
/// ([`Raise::defer`]): `v` is released where the raise ends, after the catcher's reply. Every
/// site is one no path reaches without a handler, so the raise is `Err`.
#[track_caller]
fn raise_carrying<T>(
    kind: RaiseKind,
    detail: impl fmt::Display,
    v: impl std::any::Any + Send,
) -> RaiseResult<T> {
    Err(Raise::new(kind).with_detail(detail).defer(v))
}

/// A call's value on a path where no handler of the service is live: a job, the task's once, a
/// Metal completion, the handshake. Every raise on those paths is uncaught, or passes
/// `OnFault::Die`, and died at its own site already, so the `Err` arm is never reached; were it,
/// the raise would die here.
fn unraised<T>(r: RaiseResult<T>) -> T {
    match r {
        Ok(v) => v,
        Err(r) => r.fatal(),
    }
}

/// A caught raise in a log line: its detail.
fn exception(r: &Raise) -> &str {
    r.exception()
}

/// A command buffer's error in a line: Metal's, or that Metal gave none.
fn error_text(e: Option<&crate::platform::types::NsErrorText>) -> &str {
    e.map_or("Metal gave no error", |e| e.description.as_str())
}

#[cfg(test)]
mod testing {
    //! The service's test harness: guest RAM with a task root and a one-level page table, a
    //! recording host, and the handshake.

    use std::sync::Arc;

    use parking_lot::Mutex;

    use super::GpuTask;

    use crate::guest::records::{APV_PAGE_SHIFT, GuestFeatures};
    use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
    use crate::memory::map::MemoryMap;
    use crate::platform::Handle;
    use crate::platform::device::GpuDevice;
    use crate::platform::event::SharedEvent;
    use crate::tasks::isolation::TaskLife;
    use crate::tasks::launcher::{HostLink, ServiceSide};
    use crate::tasks::wire::{Reply, TaskConfiguration, ToHost, ToService};
    use crate::testkit::guest_ram::GuestRam;
    use crate::testkit::objects::{Fk, kind};
    use crate::testkit::platform::{FAKE_REGISTRY_ID, FakePlatform};

    pub(crate) type F = FakePlatform;

    pub(crate) use crate::tasks::test_support::Lines;

    pub(super) use super::handshake::ROOT_OBJECT_LIST_LENGTH;
    pub(super) use super::memory::{PAGE, PAGE_MASK};
    pub(crate) const RAM_PA: u64 = 0x1000_0000;
    pub(crate) const RAM_LEN: u64 = 0x40_0000;
    pub(crate) const ROOT_PAGE: u32 = (RAM_PA >> APV_PAGE_SHIFT) as u32;
    pub(crate) const PT_PAGE: u32 = ROOT_PAGE + 1;
    pub(crate) const DATA_PAGE: u32 = ROOT_PAGE + 2;
    /// Four guest pages.
    pub(crate) const TASK_LEN: u64 = 4 * PAGE;
    /// A page-table entry's writable bit.
    pub(crate) const W: u32 = 0x8000_0000;

    pub(crate) fn page_pa(page: u32) -> u64 {
        u64::from(page) << APV_PAGE_SHIFT
    }

    /// The host end: records what the service sends and answers the round trips.
    #[derive(Default)]
    pub(crate) struct Host {
        pub(crate) sent: Mutex<Vec<String>>,
        pub(crate) surface: Mutex<Option<Fk<kind::Surface>>>,
        pub(crate) handle: Mutex<Option<Fk<kind::SharedTextureHandle>>>,
        /// Round trips are recorded and never answered.
        pub(crate) silent: std::sync::atomic::AtomicBool,
    }

    impl HostLink<F> for Host {
        fn send(&self, m: ToHost<F>) {
            match m {
                ToHost::RetainIoSurface { mapping_id, reply } => {
                    self.sent.lock().push(format!("retain {mapping_id}"));
                    if !self.silent.load(std::sync::atomic::Ordering::SeqCst) {
                        reply.call(self.surface.lock().as_ref().map(Handle::clone_ref));
                    }
                }
                ToHost::ReleaseIoSurface { mapping_id } => {
                    self.sent.lock().push(format!("release {mapping_id}"));
                }
                ToHost::CompleteAsyncToken {
                    token_id, success, ..
                } => {
                    self.sent
                        .lock()
                        .push(format!("complete {token_id} {success}"));
                }
                ToHost::NewSharedTextureHandle { id, reply } => {
                    self.sent.lock().push(format!("handle {id}"));
                    reply.call(self.handle.lock().as_ref().map(Handle::clone_ref));
                }
            }
        }
    }

    pub(crate) struct Tx {
        pub(crate) p: Arc<F>,
        pub(crate) ram: GuestRam,
        pub(crate) host: Arc<Host>,
        pub(crate) task: Arc<GpuTask<F>>,
        pub(crate) map: Arc<MemoryMap<F>>,
        /// The task's life: its jobs run as its code, so a death in one ends the task.
        pub(crate) life: Arc<TaskLife>,
    }

    impl Tx {
        /// Waits for the jobs queued so far.
        pub(crate) fn settle(&self) {
            self.task.queue.sync(|| ());
        }
        pub(crate) fn sels(&self) -> Vec<&'static str> {
            self.p.call_log().sels()
        }
        /// Sends `m` and waits for the jobs it queued.
        pub(crate) fn send(&self, m: ToService<F>) {
            self.task.on_message(m);
            self.settle();
        }
    }

    /// Guest RAM with a task root at `ROOT_PAGE` (depth `depth`), a one-level page table at
    /// `PT_PAGE` whose entries are `entries`, and no handshake yet.
    pub(crate) fn fixture(depth: u32, entries: &[u32]) -> Tx {
        let p = Arc::new(F::new());
        let ram = GuestRam::new();
        ram.add_range(RAM_PA, RAM_LEN).unwrap();
        p.fake_vm().adopt(&ram);
        let mut d = MemoryMapDescriptor::new();
        for r in ram.ranges() {
            d.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.region.len(),
                host_address: r.host(),
            });
        }
        let map = MemoryMap::new(p.clone(), &d).unwrap();
        ram.write_u32(page_pa(ROOT_PAGE), PT_PAGE);
        ram.write_u32(page_pa(ROOT_PAGE) + 4, depth);
        // one object list entry at task offset 0 (the handshake's set-object-list)
        ram.write_u32(page_pa(ROOT_PAGE) + ROOT_OBJECT_LIST_LENGTH, 12);
        for (i, e) in entries.iter().enumerate() {
            ram.write_u32(page_pa(PT_PAGE) + 4 * i as u64, *e);
        }
        let host = Arc::new(Host::default());
        let life = TaskLife::new();
        let task = GpuTask::new(p.clone(), host.clone(), life.clone());
        Tx {
            p,
            ram,
            host,
            task,
            map,
            life,
        }
    }

    /// `n` writable data pages from `DATA_PAGE`.
    pub(crate) fn rw_pages(n: u32) -> Vec<u32> {
        (0..n).map(|i| (DATA_PAGE + i) | W).collect()
    }

    pub(crate) type Replies<T> = Arc<Mutex<Vec<T>>>;

    /// A reply that records what it is called with.
    pub(crate) fn sink<T: Send + 'static>() -> (Replies<T>, Reply<T>) {
        let got: Replies<T> = Arc::new(Mutex::new(Vec::new()));
        let g = got.clone();
        (got, Reply::new(move |v| g.lock().push(v)))
    }

    /// The handshake's replies.
    pub(crate) fn handshake_with(
        tx: &Tx,
        task_root: u32,
        size: u64,
        registry: u64,
        map: bool,
    ) -> Vec<(u32, i32)> {
        let (got, reply) = sink();
        tx.send(ToService::Handshake {
            task_root,
            task_size: size,
            task_id: 7,
            configuration: TaskConfiguration {
                registry_id: registry,
                supports_argument_buffers: false,
            },
            features: GuestFeatures::default(),
            memory_map: map.then(|| tx.map.clone()),
            reply,
        });
        got.lock().clone()
    }

    /// A task whose handshake succeeded over `entries`.
    pub(crate) fn ready(entries: &[u32]) -> Tx {
        let tx = fixture(1, entries);
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, FAKE_REGISTRY_ID, true);
        assert_eq!(r, vec![(1, pid())]);
        tx
    }

    pub(crate) fn pid() -> i32 {
        std::process::id() as i32
    }

    /// A FIFO event for stamp index `index`, made from a real shared event's handle.
    pub(crate) fn with_fifo(tx: &Tx, index: u32) {
        let ev = tx.p.device().new_shared_event().unwrap();
        let handle = ev.new_shared_event_handle();
        tx.send(ToService::SetFifoEvent {
            event: handle,
            index,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::DeathId;
    use crate::raise::death::DeathKind as Fault;
    use crate::testkit::capture::catch_death;

    /// A fresh task has made no pipelines and rebuilt nothing; two tasks are independent
    #[test]
    fn a_fresh_task_is_zeroed() {
        use std::sync::atomic::Ordering;
        let a = testing::fixture(1, &[]);
        let b = testing::fixture(1, &[]);
        assert!(!a.task.blit.is_completed());
        assert!(!a.task.restored.load(Ordering::SeqCst));
        a.task.restored.store(true, Ordering::SeqCst);
        assert!(!b.task.restored.load(Ordering::SeqCst));
    }

    /// A raise with `Die` dies at the raise, with its kind's id; with `Return` it comes back
    /// with its detail; and a raise that reaches `unraised` dies there
    #[test]
    fn raises_die_or_come_back() {
        let d =
            catch_death(|| raise_on::<()>(OnFault::Die, RaiseKind::ZeroLength, "x")).unwrap_err();
        assert_eq!(
            (d.kind, d.file, d.line),
            (Fault::UnhandledRaise, "gpu_task.mapping.zero_length", 0)
        );
        let r = raise_on::<u32>(
            OnFault::Return,
            RaiseKind::ReplyLength,
            "reply of 0x4000 bytes",
        );
        let t = r.unwrap_err();
        assert_eq!(t.detail().unwrap_or_default(), "reply of 0x4000 bytes");
        assert_eq!((t.site().file, t.site().line), ("gpu_task.reply.length", 0));
        assert_eq!(unraised(Ok::<_, Raise>(3)), 3);
        let d = catch_death(|| unraised(Err::<(), _>(Raise::new(RaiseKind::NoBuffer))));
        assert_eq!(d.unwrap_err().file, "gpu_task.buffer.none");
    }

    /// Each death names its kind of fault and its id
    #[test]
    fn deaths_name_their_fault() {
        let cases = [
            (DeathKind::NotSetUp, Fault::NullDeref),
            (DeathKind::NullTaskRoot, Fault::NullDeref),
            (DeathKind::NullReply, Fault::NullDeref),
            (DeathKind::ShortRecords, Fault::OutOfBounds),
            (DeathKind::UnknownSurface, Fault::Abort),
            (DeathKind::WrongSurface, Fault::Abort),
            (
                DeathKind::Unsupported(Unsupported::BackingForId),
                Fault::Abort,
            ),
            (DeathKind::ZeroTaskRoot, Fault::Assert),
            (DeathKind::NoBlitLibrary, Fault::Assert),
            (DeathKind::NoComputePipeline, Fault::Assert),
        ];
        for (k, fault) in cases {
            let d = catch_death(|| -> () { crate::fatal!(k) }).unwrap_err();
            assert_eq!(d, k.site());
            assert_eq!((d.kind, d.file), (fault, k.id()));
            assert_eq!(DeathId::from(k).to_string(), k.id());
        }
        assert_eq!(RaiseKind::OutOfRange.to_string(), "gpu_task.range");
    }
}
