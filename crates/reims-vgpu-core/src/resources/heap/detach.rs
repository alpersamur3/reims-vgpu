//! The detach heap: a shared heap made over a temporary VM allocation whose backing is detached
//! at once, then replaced by the guest's address ranges at the first CPU prepare (which frees
//! the allocation). A discard detaches it again.
//!
//! A cover buffer over the whole heap answers where a buffer is expected, and is never released,
//! but after the device's teardown of every task ([`tasks_torn_down`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::guest::GuestPtr;
use crate::host::HostPtr;
use crate::num::ToU64;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Heap;
use crate::platform::types::ResourceAddressRange;
use crate::platform::values::{HeapDesc, OptionSet};
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::resources::buffer::child::HeapBufferResource;
use crate::resources::heap::HeapResourceDescriptor;
use crate::resources::resource::{
    HAZARD_TRACKING_DEFAULT, HAZARD_TRACKING_TRACKED, HEAP_TYPE_PLACEMENT,
    RESOURCE_OPTION_READ_ONLY, Resource, ResourceClass, ResourceKind, STORAGE_MODE_SHARED, TaskRef,
    features_of, platform_for, read_descriptor,
};
use crate::resources::{DeathKind, RaiseKind, raise};
use crate::rt::Hold;
use crate::sync::Mutex;

/// Ours: the platforms, by address, whose device has torn every task down ([`tasks_torn_down`]).
static TORN_DOWN: parking_lot::Mutex<Vec<usize>> = parking_lot::Mutex::new(Vec::new());

fn platform_key<P>(platform: &Arc<P>) -> usize {
    Arc::as_ptr(platform).cast::<()>().addr()
}

/// Ours: `platform`'s device tears every task down (a suspend's teardown,
/// `DeviceCore::finish_suspend`, or a stop's or reset's, `DeviceCore::teardown_tasks`). From then
/// on a detached heap that drops, made before or after, releases its cover buffer, and with it
/// the heap and its GPU VA range, whenever and however that drop comes (the last command buffer
/// that holds the heap may complete after the teardown; the guest deletes heaps after a resume);
/// before it every drop keeps the cover for good, as Metal's numbering has it. A saved state's
/// placements keep its heaps' VAs, and a resume or load in the same process, which Metal's hosts
/// never make, has the guest pin each one again, which a range still held refuses (`Metal made no
/// heap`, then every command on the heap faults): a heap the guest deleted after a save and
/// before a load of it holds one of them. The numbering past a teardown is not a fresh
/// process's anyway.
pub(crate) fn tasks_torn_down<P>(platform: &Arc<P>) {
    let key = platform_key(platform);
    let mut t = TORN_DOWN.lock();
    if !t.contains(&key) {
        t.push(key);
    }
}

/// Whether `platform`'s device has torn every task down ([`tasks_torn_down`]).
fn torn_down<P>(platform: &Arc<P>) -> bool {
    TORN_DOWN.lock().contains(&platform_key(platform))
}

/// A heap whose backing is detached until the guest's address ranges replace it.
pub struct DetachHeapResource<P: Platform> {
    task: TaskRef<P>,
    /// Never `None` after a successful creation.
    heap: Option<P::Heap>,
    /// Never released; `None` when Metal made none, which nothing checks.
    cover_buffer: Option<P::Buffer>,
    length: u64,
    page: u32,
    is_tracked: bool,
    is_read_only: bool,
    is_detached: AtomicBool,
    /// The temporary VM range the heap starts on; NULL once freed.
    dummy_allocation: Mutex<HostPtr>,
    /// The platform whose VM holds the temporary allocation.
    platform: Arc<P>,
}

impl<P: Platform> DetachHeapResource<P> {
    /// A detached heap of the record's size. The private bit is ignored; a tracked heap needs
    /// hazard tracking and a read-only one argument buffers, in that order (logged, `None`). A
    /// guest pin is passed to Metal; a zero slot gets the made address.
    /// A raise after the heap is made leaks it.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let platform = platform_for(&task);
        let d: HeapResourceDescriptor = read_descriptor(descriptor);
        let is_tracked = d.is_tracked();
        let length = d.length();
        if is_tracked && !features_of(&task).supports_heap_hazard_tracking {
            log::warn!("tracked heap refused: the task's features have no heap hazard tracking");
            return Ok(None);
        }
        let is_read_only = d.is_read_only();
        if is_read_only && !features_of(&task).supports_argument_buffers {
            log::warn!("read-only heap refused: the task's features have no argument buffers");
            return Ok(None);
        }
        // The descriptor, the VM and the range array need the platform.
        let Some(p) = platform else {
            return Ok(None);
        };
        let mut options = vec![OptionSet::StorageMode(STORAGE_MODE_SHARED)];
        if is_read_only {
            options.push(OptionSet::ResourceOptionsOr(RESOURCE_OPTION_READ_ONLY));
        }
        // A failure allocates nothing, so nothing is freed. Metal's heap is made over its
        // pages, which the platform maps again as an alias: Mach's `mach_vm_allocate`.
        let dummy = match p.vm().allocate_aliasable(length) {
            Ok(a) => a,
            Err(error) => {
                log::error!(
                    "no address space for a detached heap of {length:#x} bytes: kern_return {}",
                    error.0
                );
                return Ok(None);
            }
        };
        // From here a refusal frees the allocation, and a raise leaks it.
        let range = ResourceAddressRange {
            address: dummy.addr().to_u64(),
            length,
        };
        let ranges = p.new_address_range_array(&[range])?;
        options.push(OptionSet::HazardTrackingMode(if is_tracked {
            HAZARD_TRACKING_TRACKED
        } else {
            HAZARD_TRACKING_DEFAULT
        }));
        // A guest pin (the first of two reads of the slot).
        let pin = placement
            .map(|pl| pl.read_u64(0, DeathKind::NullPlacement))
            .filter(|pin| *pin != 0);
        let desc = HeapDesc {
            size: length,
            heap_type: HEAP_TYPE_PLACEMENT,
            options,
            address_ranges: Some(&ranges),
            pinned_gpu_address: pin,
        };
        let made = match task.upgrade() {
            Some(t) => t
                .mtl_device()
                .map_or(Ok(None), |device| device.new_heap(&desc).map(Some))?,
            None => None,
        };
        let (heap, descriptor_options) =
            made.map_or((None, 0), |m| (m.heap, m.descriptor_resource_options));
        // Only into a zero slot, before the nil check (a failed creation writes 0; a pin is kept).
        if let Some(pl) = placement
            && pl.read_u64(0, DeathKind::NullPlacement) == 0
        {
            let address = heap.as_ref().map_or(0, Heap::gpu_address);
            pl.write_u64(0, address, DeathKind::NullPlacement);
        }
        let Some(heap) = heap else {
            log::error!("Metal made no heap over the detached heap's {length:#x} bytes");
            // Freed only when non-zero: XNU gives a zero-length allocation address 0.
            if !dummy.is_null() {
                let _ = p.vm().deallocate(dummy, length);
            }
            return Ok(None);
        };
        // A raise from here on keeps the heap for good: its GPU VA is already in the guest's
        // placement, and the VAs later heaps get and the pins they keep follow from it.
        if let Err(e) = heap.detach_backing() {
            keep_forever(heap, Kept::MetalNumbering);
            return Err(e);
        }
        // At the descriptor's options as Metal reports them; the buffer is not checked for nil.
        let cover_buffer = match heap.new_buffer_of_length(length, descriptor_options, 0) {
            Ok(b) => b,
            Err(e) => {
                keep_forever(heap, Kept::MetalNumbering);
                return Err(e);
            }
        };
        Ok(Some(Resource::new(ResourceKind::DetachHeap(
            DetachHeapResource {
                task,
                heap: Some(heap),
                cover_buffer,
                length,
                page: d.page(),
                is_tracked,
                is_read_only,
                is_detached: AtomicBool::new(true),
                dummy_allocation: Mutex::new(dummy),
                platform: p,
            },
        ))))
    }

    /// The task.
    pub fn task(&self) -> &TaskRef<P> {
        &self.task
    }

    /// The record's tracked and read-only bits.
    pub fn flags(&self) -> (bool, bool) {
        (self.is_tracked, self.is_read_only)
    }

    fn detached(&self) -> bool {
        self.is_detached.load(Ordering::Relaxed)
    }

    /// Replaces a detached backing with the guest's address ranges, then frees the temporary
    /// allocation once. Metal's refusal (or a torn-down task's missing ranges) raises and keeps
    /// it detached.
    fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        if !self.detached() {
            return Ok(());
        }
        let mut read_only = false;
        let ranges = match self.task.upgrade() {
            Some(t) => {
                t.address_ranges_for_virtual_page(hold, self.page, self.length, &mut read_only)?
            }
            None => None,
        };
        let replaced = match (&self.heap, &ranges) {
            (Some(h), Some(r)) => h.replace_backing(r, read_only)?,
            (None, _) => false,
            // As the detach buffer's: only a torn-down task gives no ranges.
            (Some(_), None) => false,
        };
        if !replaced {
            return raise(RaiseKind::HeapBackingRefused);
        }
        self.is_detached.store(false, Ordering::Relaxed);
        // Freed once; a later discard does not allocate it again.
        let mut dummy = self.dummy_allocation.lock();
        if !dummy.is_null() {
            let _ = self.platform.vm().deallocate(*dummy, self.length);
            *dummy = HostPtr::NULL;
        }
        Ok(())
    }
}

impl<P: Platform> ResourceClass<P> for DetachHeapResource<P> {
    fn initial_guest_valid(&self) -> bool {
        true
    }

    fn get_heap(&self, this: &Resource<P>) -> RaiseResult<Option<P::Heap>> {
        let _ = this;
        Ok(self.heap.as_ref().map(Handle::clone_ref))
    }

    /// The cover buffer: a heap listed where a buffer is expected.
    fn get_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Buffer>> {
        let _ = this;
        Ok(self.cover_buffer.as_ref().map(Handle::clone_ref))
    }

    fn needs_cpu_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn prepare_with_cpu(&self, hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.ensure_physical(hold)
    }

    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.heap.as_ref().map_or(Ok(()), |h| h.detach_backing())?;
        self.is_detached.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn needs_sync(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        // Derived from the backing; the host flag is dead.
        !self.detached()
    }

    fn set_is_guest_valid(&self, this: &Resource<P>, v: bool) {
        let _ = (this, v);
    }

    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn new_child_buffer_resource_with_buffer(
        &self,
        this: &Arc<Resource<P>>,
        buffer: P::Buffer,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        // Discardable.
        Ok(Some(HeapBufferResource::new(this.clone(), buffer, true)))
    }
}

impl<P: Platform> Drop for DetachHeapResource<P> {
    fn drop(&mut self) {
        // The cover buffer is never released, and with it (Metal's sub-allocation retain) the heap
        // and its GPU VA range; but once the device has torn every task down.
        if let Some(b) = self.cover_buffer.take() {
            if torn_down(&self.platform) {
                drop(b);
            } else {
                keep_forever(b, Kept::MetalNumbering);
            }
        }
        drop(self.heap.take());
        let dummy = *self.dummy_allocation.get_mut();
        if !dummy.is_null() {
            let _ = self.platform.vm().deallocate(dummy, self.length);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::os::vm::KernReturn;
    use crate::resources::Op;
    use crate::resources::heap::tests::heap_desc;
    use crate::resources::resource::test_support::{Env, gone_task, kind, logged, placement};
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::{Fk, Model, kind};
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    fn desc(page: u32, length: u64, flags: u8) -> Vec<u8> {
        heap_desc(page, length, flags)
    }

    fn heap(env: &Env, flags: u8, pl: Option<GuestPtr>) -> Option<Arc<Resource<Fp>>> {
        env.run(|| DetachHeapResource::new(env.task(), &desc(4, 0x8000, flags), pl))
            .unwrap()
    }

    fn parts(r: &Resource<Fp>) -> &DetachHeapResource<Fp> {
        match &r.kind {
            ResourceKind::DetachHeap(k) => k,
            _ => panic!("kind"),
        }
    }

    fn dummy_of(env: &Env) -> usize {
        match env.p.call_log().calls_to("initWithRanges:count:")[0].args[0] {
            Arg::Val(ref s) => {
                let hex = s
                    .split("address: ")
                    .nth(1)
                    .and_then(|x| x.split(',').next());
                hex.and_then(|x| x.trim().parse::<usize>().ok())
                    .expect("address")
            }
            ref x => panic!("{x:?}"),
        }
    }

    /// The value the last heap was made from, as the creator logged it.
    fn made_from(env: &Env) -> String {
        match env.p.call_log().calls_to("newHeapWithDescriptor:").last() {
            Some(c) => match &c.args[0] {
                Arg::Val(text) => text.clone(),
                x => panic!("{x:?}"),
            },
            None => panic!("no heap made"),
        }
    }

    fn provide_ranges(env: &Env, page: u32) -> Fk<kind::AddressRangeArray> {
        let a = env
            .p
            .world()
            .new_obj::<kind::AddressRangeArray>(Model::None);
        env.rec.provide(u64::from(page), &a);
        a
    }

    mod init {
        use super::*;

        #[test]
        fn makes_the_heap_over_a_dummy_allocation() {
            let env = Env::new();
            let r = heap(&env, 0, None).unwrap();
            let log = env.p.call_log();
            assert_eq!(
                log.sels(),
                [
                    "mach_vm_allocate",
                    "initWithRanges:count:",
                    "mtl_device",
                    "newHeapWithDescriptor:",
                    "detachBacking",
                    "newBufferWithLength:options:offset:",
                ]
            );
            // A shared placement heap over the dummy's range, untracked, with no pin.
            let made = made_from(&env);
            assert!(
                made.starts_with(
                    "HeapDesc { size: 32768, heap_type: 1, options: [StorageMode(0), \
                     HazardTrackingMode(0)], address_ranges: Some("
                ),
                "{made}"
            );
            assert!(made.ends_with("pinned_gpu_address: None }"), "{made}");
            assert_eq!(
                log.calls_to("mach_vm_allocate")[0].args[0],
                Arg::U64(0x8000)
            );
            assert_eq!(
                log.calls_to("newBufferWithLength:options:offset:")[0].args,
                vec![Arg::U64(0x8000), Arg::U64(0), Arg::U64(0)]
            );
            assert!(r.state.guest_valid() && !r.state.host_valid());
            assert!(r.needs_prepare() && !r.is_host_valid());
            assert!(!parts(&r).dummy_allocation.lock().is_null());
            assert_eq!(parts(&r).flags(), (false, false));
        }

        #[test]
        fn tracked_and_read_only_need_their_features_in_that_order() {
            let env = Env::new();
            let (made, lines) = logged(|| heap(&env, 6, None));
            assert!(made.is_none());
            assert_eq!(
                lines,
                vec!["tracked heap refused: the task's features have no heap hazard tracking"]
            );
            env.set_features(|f| f.supports_heap_hazard_tracking = true);
            let (made, lines) = logged(|| heap(&env, 6, None));
            assert!(made.is_none());
            assert_eq!(
                lines,
                vec!["read-only heap refused: the task's features have no argument buffers"]
            );
            assert_eq!(
                env.p.call_log().count("mach_vm_allocate"),
                0,
                "before the allocation"
            );
            env.set_features(|f| {
                f.supports_heap_hazard_tracking = true;
                f.supports_argument_buffers = true;
            });
            let r = heap(&env, 6, None).unwrap();
            assert_eq!(parts(&r).flags(), (true, true));
            let log = env.p.call_log();
            assert!(made_from(&env).contains(
                "options: [StorageMode(0), ResourceOptionsOr(16777216), HazardTrackingMode(2)]"
            ));
            // The cover buffer takes the options read back from the descriptor.
            assert_eq!(
                log.calls_to("newBufferWithLength:options:offset:")[0].args[1],
                Arg::U64(0x100_0000)
            );
        }

        #[test]
        fn tracked_on_a_gone_task_dies_reading_its_features() {
            let d =
                catch_death(|| DetachHeapResource::<Fp>::new(gone_task(), &desc(1, 16, 2), None));
            assert_eq!(d.err(), Some(DeathKind::GoneTaskFeatures.site()));
        }

        #[test]
        fn a_failed_allocation_logs_and_makes_nothing() {
            let env = Env::new();
            env.p
                .script()
                .push("mach_vm_allocate", Step::value(KernReturn(3)));
            let (made, lines) = logged(|| heap(&env, 0, None));
            assert!(made.is_none());
            assert_eq!(
                lines,
                vec!["no address space for a detached heap of 0x8000 bytes: kern_return 3"]
            );
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 0);
            assert_eq!(env.p.call_log().count("newHeapWithDescriptor:"), 0);
        }

        #[test]
        fn a_pin_is_passed_and_kept_even_when_creation_fails() {
            // The suite's `mutgap2_res_heap_pinned_huge` measures the same.
            let env = Env::new();
            let (m, pl) = placement(0x2_0000_0000);
            env.p.script().push("newHeapWithDescriptor:", Step::Nil);
            assert!(heap(&env, 0, Some(pl)).is_none());
            assert!(made_from(&env).contains("pinned_gpu_address: Some(8589934592)"));
            assert_eq!(m.read_u64(0), Some(0x2_0000_0000));
            // The release frees the dummy allocation.
            let dummy = dummy_of(&env);
            let c = env.p.call_log().calls_to("mach_vm_deallocate");
            assert_eq!(c[0].args, vec![Arg::Ptr(dummy), Arg::U64(0x8000)]);
        }

        #[test]
        fn a_nil_heap_frees_only_a_non_zero_dummy() {
            // The free is guarded by a non-zero address. XNU returns address 0 for a zero
            // length, so no mach_vm_deallocate is made; the fake still returns a mapped page for
            // length 0, so this asserts the guard in the form that holds for either answer.
            let env = Env::new();
            env.p.script().push("newHeapWithDescriptor:", Step::Nil);
            let r = env
                .run(|| DetachHeapResource::new(env.task(), &desc(1, 0, 0), None))
                .unwrap();
            assert!(r.is_none());
            let dummy = dummy_of(&env);
            let c = env.p.call_log().calls_to("mach_vm_deallocate");
            if dummy == 0 {
                assert!(c.is_empty());
            } else {
                assert_eq!(c.len(), 1);
                assert_eq!(c[0].args, vec![Arg::Ptr(dummy), Arg::U64(0)]);
            }
        }

        #[test]
        fn a_zero_entry_gets_the_heap_address() {
            let env = Env::new();
            let (m, pl) = placement(0);
            let r = heap(&env, 0, Some(pl)).unwrap();
            let h = r.get_heap().unwrap().unwrap();
            assert_ne!(h.gpu_address(), 0);
            assert_eq!(m.read_u64(0), Some(h.gpu_address()));
        }

        #[test]
        fn a_nil_cover_buffer_is_not_checked() {
            let env = Env::new();
            env.p
                .script()
                .push("newBufferWithLength:options:offset:", Step::Nil);
            let r = heap(&env, 0, None).unwrap();
            assert!(r.get_buffer().unwrap().is_none());
        }

        #[test]
        fn a_heap_raise_propagates_and_leaks_the_dummy_allocation() {
            let env = Env::new();
            env.p.script().push(
                "newHeapWithDescriptor:",
                Step::raise("NSInvalidArgumentException", "x"),
            );
            assert!(
                env.run(|| DetachHeapResource::new(env.task(), &desc(4, 0x4000, 0), None))
                    .is_err()
            );
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 0);
        }
    }

    mod paging {
        use super::*;

        #[test]
        fn prepare_replaces_the_backing_and_frees_the_dummy_once() {
            let env = Env::new();
            let r = heap(&env, 0, None).unwrap();
            let dummy = dummy_of(&env);
            let ranges = provide_ranges(&env, 4);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert_eq!(
                env.rec.log().calls_to("address_ranges_for_virtual_page")[0].args,
                vec![Arg::U32(4), Arg::U64(0x8000), Arg::Bool(false)]
            );
            let rep = env
                .p
                .call_log()
                .calls_to("replaceBackingWithRanges:readOnly:");
            assert_eq!(rep[0].args, vec![Arg::obj(&ranges), Arg::Bool(false)]);
            let c = env.p.call_log().calls_to("mach_vm_deallocate");
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].args, vec![Arg::Ptr(dummy), Arg::U64(0x8000)]);
            assert!(r.is_host_valid() && !r.needs_prepare());
            // discard detaches; the next prepare rebinds and frees nothing more.
            r.discard().unwrap();
            assert!(!r.is_host_valid() && r.needs_prepare());
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 1);
            assert_eq!(env.p.call_log().count("detachBacking"), 2);
            let m = env.p.call_log().mark();
            r.synchronize_with_cpu().unwrap();
            assert!(env.p.call_log().since(m).is_empty());
            // The last release: nothing left to free.
            drop(r);
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 1);
        }

        #[test]
        fn a_refused_replacement_raises_and_keeps_the_dummy() {
            let env = Env::new();
            let r = heap(&env, 0, None).unwrap();
            provide_ranges(&env, 4);
            env.p
                .script()
                .push("replaceBackingWithRanges:readOnly:", Step::value(false));
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_with_cpu(hold)),
                Some(RaiseKind::HeapBackingRefused)
            );
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 0);
            // The last release frees it.
            drop(r);
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 1);
        }
    }

    mod accessors {
        use super::*;

        #[test]
        fn get_buffer_is_the_cover_buffer() {
            let env = Env::new();
            let r = heap(&env, 0, None).unwrap();
            let b = r.get_buffer().unwrap().unwrap();
            assert_eq!(
                Some(b.id()),
                parts(&r).cover_buffer.as_ref().map(|x| x.id())
            );
            assert!(r.get_heap().unwrap().is_some());
            assert!(r.needs_cpu_paging() && !r.needs_sync() && r.supports_discard());
            r.set_is_guest_valid(false);
            assert!(r.is_guest_valid() && r.state.guest_valid());
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.get_resource()),
                Some(RaiseKind::Unrecognized(Op::GetResource))
            );
            assert_eq!(
                kind(r.prepare_in_encoder(hold, None)),
                Some(RaiseKind::Unrecognized(Op::Prepare))
            );
        }

        #[test]
        fn children_are_discardable() {
            let env = Env::new();
            let r = heap(&env, 0, None).unwrap();
            let h = r.get_heap().unwrap().unwrap();
            let b = h.new_buffer_of_length(16, 0, 0).unwrap().unwrap();
            let c = r.new_child_buffer_resource_with_buffer(b).unwrap().unwrap();
            assert!(c.supports_discard());
        }

        #[test]
        fn last_release_leaks_the_cover_buffer_and_frees_the_dummy() {
            // The cover buffer is never released, and so (by Metal's sub-allocation retain) neither
            // is the heap's GPU VA range.
            let env = Env::new();
            let r = heap(&env, 0, None).unwrap();
            let cover = r.get_buffer().unwrap().unwrap();
            let h = r.get_heap().unwrap().unwrap();
            let (c0, h0) = (cover.retain_count(), h.retain_count());
            drop(r);
            assert_eq!(cover.retain_count(), c0, "never released");
            assert_eq!(h.retain_count(), h0 - 1, "the heap is released");
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 1);
        }

        /// Ours: once its platform's device has torn every task down, the last release releases
        /// the cover buffer too, of a heap made before the teardown or after it; a heap of another
        /// platform still keeps its own.
        #[test]
        fn a_teardown_releases_the_cover_buffer() {
            let env = Env::new();
            let other = Env::new();
            let r = heap(&env, 0, None).unwrap();
            let o = heap(&other, 0, None).unwrap();
            let cover = r.get_buffer().unwrap().unwrap();
            let other_cover = o.get_buffer().unwrap().unwrap();
            let (c0, o0) = (cover.retain_count(), other_cover.retain_count());
            tasks_torn_down(&env.p);
            let after = heap(&env, 0, None).unwrap();
            let after_cover = after.get_buffer().unwrap().unwrap();
            let a0 = after_cover.retain_count();
            drop(r);
            drop(o);
            drop(after);
            assert_eq!(cover.retain_count(), c0 - 1, "made before: released");
            assert_eq!(after_cover.retain_count(), a0 - 1, "made after: released");
            assert_eq!(other_cover.retain_count(), o0, "another platform's kept");
        }
    }
}
