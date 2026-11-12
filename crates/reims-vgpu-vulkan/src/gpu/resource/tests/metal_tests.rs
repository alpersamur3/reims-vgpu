//! The tests of `resource` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::types::{MtlOrigin, SizeAndAlign};
use reims_vgpu_core::platform::values::{
    BufferDesc, HeapDesc, HeapMade, OptionSet, SamplerDesc, TextureDesc,
};
use reims_vgpu_metal::MetalPlatform;

use super::*;
use crate::os::OsSurface;

/// Of the suite's kernel function, made on `d`, a state without indirect command buffer
/// support and one with it (`supportIndirectCommandBuffers`, compute key 7, set): whether
/// each supports them, its `resourceIndex` and its `gpuResourceID`.
fn kernel_states<P: Platform>(d: &P::Device) -> [(bool, u64, u64); 2] {
    use reims_vgpu_core::platform::resource::{ComputePipeline as _, Library as _};
    use reims_vgpu_core::rt::Hold;
    let kernel = include_bytes!("../../../../../../tests/suite/fixtures/shaders/kernel.metallib");
    let library = d.new_library(kernel).unwrap().unwrap().unwrap();
    let name = library.function_names()[0].clone();
    let f = library
        .new_function(OnFault::Return, &name)
        .unwrap()
        .unwrap();
    let plain = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    // Two entries: key 7 (`supportIndirectCommandBuffers`) 1, and key 0, the function's ref.
    let blob = [2, 7, 4, 1, 0, 0, 0, 0, 4, 0x11, 0, 0, 0];
    let mut resolve = |_: Hold<'_>, _: u64| Ok(Some(f.clone_ref()));
    let desc = d
        .deserialize_compute_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    let supporting = d
        .new_compute_pipeline_state(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap();
    [&plain, &supporting].map(|p| {
        (
            p.support_indirect_command_buffers(),
            p.resource_index(),
            p.gpu_resource_id(),
        )
    })
}

/// A compute state's `resourceIndex` is 0 without indirect command buffer support and its
/// `gpuResourceID` with it, as Metal's is
#[test]
fn a_compute_states_resource_index_is_metals() {
    let shape = |[plain, supporting]: [(bool, u64, u64); 2]| {
        (
            plain,
            supporting.0,
            supporting.1 == supporting.2,
            supporting.2 != 0,
        )
    };
    let ours = shape(kernel_states::<VulkanPlatform>(&dev()));
    assert_eq!(ours, ((false, 0, 0), true, true, true));
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(shape(kernel_states::<MetalPlatform>(&md)), ours);
    }
}

/// newBufferWithIOSurface: makes a Shared buffer over the surface's memory that holds one
/// reference of the surface while it lives, whose contents are NULL, as Metal's; iosurface
/// answers it, +1
#[test]
fn buffers_on_surfaces() {
    let d = dev();
    let s = bgra_surface(64, 32);
    assert_eq!(s.retain_count(), 1);
    let b = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(s.retain_count(), 2);
    let metal = MetalPlatform::system_default_device().unwrap();
    let other = bgra_surface(64, 32);
    let on_metal = metal.new_buffer_on_surface(Some(&other)).unwrap().unwrap();
    let seen = (b.contents(), b.length(), b.storage_mode());
    assert_eq!(seen, (HostPtr::NULL, s.alloc_size(), 0));
    assert_eq!(
        seen,
        (
            on_metal.contents(),
            on_metal.length(),
            on_metal.storage_mode()
        )
    );
    drop((on_metal, other));
    let again = b.iosurface().unwrap();
    assert_eq!(again.id(), s.id());
    assert_eq!(s.retain_count(), 3);
    drop(again);
    drop(b);
    assert_eq!(s.retain_count(), 1);
}

/// Every buffer over a surface answers the surface's address, as on Metal (where it runs, two
/// live ones answer one address, and so does one made after the first's release while a
/// command buffer that fills it waits): a buffer made while another holds the address shares
/// a live one's range, which stays while one holds it. One made while none lives, after the
/// last's release while a command buffer that fills it waits, dies at
/// newBufferWithIOSurface:, naming the surface, as the runtime holds the released one's range
/// until that command buffer completed and shares a range with a live buffer alone. Then a
/// buffer over the surface lies at the address again
#[test]
fn buffers_over_a_surface_share_its_address() {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    fn waiting_fill<P: Platform>(
        d: &P::Device,
        buffer: &P::Buffer,
    ) -> (P::CommandBuffer, P::SharedEvent) {
        let q = d.new_command_queue().unwrap();
        let gate = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&gate)), 1).unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer,
            range: NsRange::new(0, 16),
            value: 0x5c,
        };
        e.encode(OnFault::Return, fill).unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        (cb, gate)
    }
    fn completed<P: Platform>((cb, gate): (P::CommandBuffer, P::SharedEvent)) {
        gate.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    }
    if let Some(metal) = MetalPlatform::system_default_device() {
        let other = bgra_surface(64, 64);
        let first = metal.new_buffer_on_surface(Some(&other)).unwrap().unwrap();
        let second = metal.new_buffer_on_surface(Some(&other)).unwrap().unwrap();
        let at = first.gpu_address();
        assert_eq!(second.gpu_address(), at, "Metal");
        drop(second);
        let waiting = waiting_fill::<MetalPlatform>(&metal, &first);
        drop(first);
        let third = metal.new_buffer_on_surface(Some(&other)).unwrap().unwrap();
        assert_eq!(
            third.gpu_address(),
            at,
            "Metal, the first released, its fill waiting"
        );
        completed::<MetalPlatform>(waiting);
    }
    let d = dev();
    let s = bgra_surface(64, 64);
    let first = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    let at = first.gpu_address();
    let second = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(second.gpu_address(), at, "the first live");
    drop(first);
    let third = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(
        third.gpu_address(),
        at,
        "the first released, the second live"
    );
    drop(third);
    let waiting = waiting_fill::<VulkanPlatform>(&d, &second);
    drop(second);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| drop(d.new_buffer_on_surface(Some(&s)))).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newBufferWithIOSurface:"),
        "{line}"
    );
    let named = format!(
        "a buffer over IOSurface {:#x} while a released one",
        s.id().0
    );
    assert!(line.contains(&named), "{line}");
    completed::<VulkanPlatform>(waiting);
    let again = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(again.gpu_address(), at, "made once the fill completed");
}

/// A 2D array over a no-copy buffer of host memory made by
/// `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` is Metal's, byte for
/// byte: its slices as made, after the CPU rewrote the memory (both alias it), and the memory
/// after a blit copied into every slice (where its slices lie). Metal's is measured first
#[test]
fn array_over_host_memory_is_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let metal = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let theirs = array_over_host_memory_on::<MetalPlatform>(&md, &metal, true);
    assert!(theirs[0] != theirs[1], "Metal's aliases the memory");
    let ours = array_over_host_memory_on::<VulkanPlatform>(
        &dev(),
        &|t| platform().texture_descriptor(t),
        true,
    );
    for (step, (ours, theirs)) in ["made", "rewritten", "after"]
        .iter()
        .zip(ours.iter().zip(&theirs))
    {
        assert!(ours == theirs, "{step}");
    }
}

/// newBufferWithIOSurface: of a surface over the guest's pages at an offset into its
/// allocation (`kIOSurfaceOffset`) reads as Metal's does: as long as the allocation, from the
/// surface's base address, its last bytes, past the allocation's end, 0
#[test]
fn buffers_on_offset_surfaces_read_as_metals() {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::resource::Library as _;
    // `copy_words` of the surface's buffer, on `d`: its length, and each 32-bit word of it.
    fn copied<P: Platform<Surface = OsSurface>>(d: &P::Device, s: &OsSurface) -> (u64, Vec<u32>) {
        let b = d.new_buffer_on_surface(Some(s)).unwrap().unwrap();
        let length = b.length();
        let library = d
            .new_library(crate::gpu::ycbcr::measure::PROBE)
            .unwrap()
            .unwrap()
            .unwrap();
        let f = library
            .new_function(OnFault::Die, "copy_words")
            .unwrap()
            .unwrap();
        let copies = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let out = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&b), Some(&out)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&copies),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(length / 4, 1, 1),
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let n = usize::try_from(length / 4).unwrap();
        // SAFETY: a Shared buffer's words, read after the command buffer completed.
        let words = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), n) };
        (length, words.to_vec())
    }
    for offset in [0, 0x40, 0x1000, 0x4000] {
        let (s, pages, length) = crate::os::tests::bgra_surface_over_at(64, 32, offset);
        // Each word of the pages holds its own offset there, its top bit set.
        for at in (0..length).step_by(4) {
            // SAFETY: a word of the test's pages, which no GPU work uses now.
            unsafe {
                pages
                    .as_ptr::<u8>()
                    .add(usize::try_from(at).unwrap())
                    .cast::<u32>()
                    .write(0x8000_0000 | u32::try_from(at).unwrap());
            }
        }
        // Metal's first, before this platform's runtime makes any buffer of its own.
        let metal =
            MetalPlatform::system_default_device().map(|md| copied::<MetalPlatform>(&md, &s));
        let (ours, words) = copied::<VulkanPlatform>(&dev(), &s);
        assert_eq!(ours, length, "{offset:#x}: the allocation's size");
        let past = usize::try_from((length - offset) / 4).unwrap();
        assert_eq!(
            words[0],
            0x8000_0000 | u32::try_from(offset).unwrap(),
            "{offset:#x}"
        );
        assert!(
            words[past..].iter().all(|&w| w == 0),
            "{offset:#x}: 0 past the allocation"
        );
        if let Some(metal) = metal {
            assert!(metal == (ours, words), "{offset:#x}: Metal's");
        }
        drop(s);
        crate::os::OsVm::new().deallocate(pages, length);
    }
}

/// The answers of setPurgeableState: to KeepCurrent, Volatile, NonVolatile, Empty,
/// KeepCurrent and NonVolatile in turn, of each of `P`'s resources: Shared buffers of 256
/// bytes and 16 KiB; Shared textures of 8 by 8 and 256 by 256 BGRA8Unorm, a view of the larger,
/// a texture over the larger buffer; and Private shareable textures of 8 by 8 and 256 by 256.
fn purgeable_answers_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<Vec<u64>>
where
    P::Device: GpuDevice<P>,
{
    use reims_vgpu_core::platform::resource::{Buffer as _, GpuResource, Texture as _};
    let states = [1, 3, 2, 4, 1, 2];
    let answers = |r: &dyn GpuResource<P>| {
        states
            .iter()
            .map(|&s| r.set_purgeable_state(s).unwrap())
            .collect::<Vec<_>>()
    };
    let texture = |side: u64, options: u64| {
        desc(&TextureDesc {
            pixel_format: 80,
            width: side,
            height: side,
            usage: 0x13,
            resource_options: Some(options),
            ..TextureDesc::default()
        })
    };
    let buffer = |length| {
        d.new_buffer_of_length(OnFault::Return, length, 0)
            .unwrap()
            .unwrap()
    };
    let (small, large) = (buffer(256), buffer(16384));
    let small_texture = d.new_texture(OnFault::Return, &texture(8, 0)).unwrap();
    let large_texture = d.new_texture(OnFault::Return, &texture(256, 0)).unwrap();
    let (small_texture, large_texture) = (small_texture.unwrap(), large_texture.unwrap());
    let view = large_texture.new_texture_view(81).unwrap().unwrap();
    let over = large
        .new_texture(OnFault::Return, &texture(8, 0), 0, 256)
        .unwrap()
        .unwrap();
    let shared_small = d.new_shared_texture(&texture(8, 0x20)).unwrap().unwrap();
    let shared_large = d.new_shared_texture(&texture(256, 0x20)).unwrap().unwrap();
    vec![
        answers(&small),
        answers(&large),
        answers(&small_texture),
        answers(&large_texture),
        answers(&view),
        answers(&over),
        answers(&shared_small),
        answers(&shared_large),
    ]
}

/// setPurgeableState: answers each resource as the Metal platform does on this host: the
/// state before for a buffer of 16 KiB, a texture of its own allocation of 16 KiB or more and
/// a shareable texture; NonVolatile to every request for a smaller buffer or texture, a view
/// and a texture over a buffer
#[test]
fn purgeable_states_are_the_metal_platforms() {
    let ours =
        purgeable_answers_on::<VulkanPlatform>(&dev(), &|t| platform().texture_descriptor(t));
    let (kept, non_volatile) = ([2, 2, 3, 2, 4, 4], [2; 6]);
    let expected = [
        non_volatile,
        kept,
        non_volatile,
        kept,
        non_volatile,
        non_volatile,
        kept,
        kept,
    ];
    assert_eq!(ours, expected);
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    assert_eq!(ours, purgeable_answers_on::<MetalPlatform>(&metal, &desc));
}

/// A colour texture's heap size and alignment are the Metal platform's on this host, with
/// allowGPUOptimizedContents on, off and unset (a depth or stencil texture's:
/// `heap_depth_stencil_sizes_with_optimized_contents_off`), and so are a two-plane XR10
/// texture's (550, 551), whose two planes Metal sizes: 2D of one and seven levels, at a
/// display's size, and a 2D array
#[test]
fn heap_texture_sizes_are_the_metal_platforms() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    for pixel_format in [550, 551] {
        for (texture_type, width, height, mipmap_level_count, array_length) in [
            (2, 64, 64, 1, 1),
            (2, 64, 64, 7, 1),
            (2, 1920, 1080, 1, 1),
            (3, 256, 256, 1, 4),
        ] {
            let t = TextureDesc {
                texture_type,
                pixel_format,
                width,
                height,
                mipmap_level_count,
                array_length,
                usage: 0x1,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            };
            let ours = dev()
                .heap_texture_size_and_align(&platform().texture_descriptor(&t))
                .unwrap();
            let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&t);
            let theirs = metal.heap_texture_size_and_align(&desc).unwrap();
            assert_eq!(ours, theirs, "{t:?}");
        }
    }
    for (pixel_format, width, height, mipmap_level_count, usage, storage) in [
        (70, 256, 256, 1, 0x1, 0x20),
        (80, 64, 32, 7, 0x13, 0x20),
        (80, 100, 100, 1, 0x5, 0),
        (115, 33, 17, 1, 0x3, 0x20),
    ] {
        for allow_gpu_optimized_contents in [None, Some(true), Some(false)] {
            let t = TextureDesc {
                pixel_format,
                width,
                height,
                mipmap_level_count,
                usage,
                resource_options: Some(storage),
                allow_gpu_optimized_contents,
                ..TextureDesc::default()
            };
            let ours = dev()
                .heap_texture_size_and_align(&platform().texture_descriptor(&t))
                .unwrap();
            let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&t);
            let theirs = metal.heap_texture_size_and_align(&desc).unwrap();
            assert_eq!(ours, theirs, "{t:?}");
        }
    }
}

/// A 2- or 4-sample texture's heap size and alignment are the Metal platform's on this host
/// (the runtime answers Metal's rule): 2D
/// multisample textures and multisample arrays of one and three slices, of colour formats of
/// 4 and 8 bytes and of the depth and stencil formats, from 4 by 4 to 256 by 256, render
/// targets read by shaders or not, and colour ones of `PixelFormatView` usage or with
/// `allowGPUOptimizedContents` off, which Metal keeps uncompressed. Each shape whose answer
/// differs, or whose query dies, is named
#[test]
fn multisample_heap_texture_sizes_are_the_metal_platforms() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let (p, d) = (platform(), dev());
    let colour = [70, 80, 115];
    let depth = [252, 253, 260];
    let mut shapes = Vec::new();
    for pixel_format in colour.into_iter().chain(depth) {
        let ways: &[(u64, Option<bool>)] = match colour.contains(&pixel_format) {
            true => &[(0x4, None), (0x5, None), (0x14, None), (0x5, Some(false))],
            false => &[(0x4, None), (0x5, None)],
        };
        for sample_count in [2, 4] {
            for (texture_type, array_length) in [(4, 1), (8, 1), (8, 3)] {
                for (width, height) in [(4, 4), (16, 8), (16, 16), (100, 60), (256, 256)] {
                    for &(usage, allow_gpu_optimized_contents) in ways {
                        shapes.push(TextureDesc {
                            texture_type,
                            pixel_format,
                            width,
                            height,
                            sample_count,
                            array_length,
                            usage,
                            resource_options: Some(0x20),
                            allow_gpu_optimized_contents,
                            ..TextureDesc::default()
                        });
                    }
                }
            }
        }
    }
    let mut differ = Vec::new();
    for t in &shapes {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(t);
        let theirs = metal.heap_texture_size_and_align(&desc).unwrap();
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| {
            d.heap_texture_size_and_align(&p.texture_descriptor(t))
                .unwrap()
        });
        match ours {
            Ok(ours) if ours == theirs => {}
            Ok(ours) => differ.push(format!("{t:?}: ours {ours:?}, Metal's {theirs:?}")),
            Err(site) => {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                differ.push(format!("{t:?}: died at {site:?}: {line}"));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {}: {differ:#?}",
        differ.len(),
        shapes.len()
    );
}

/// Depth and stencil textures without allowGPUOptimizedContents, private, read by shaders and
/// rendered to, take what Metal's of their shape take of its space, though the runtime's
/// image is larger (`metal_space::own_heap_size`): on Metal and here alike, how far the next
/// range moves as one is made, and how far a private buffer of 0x100 bytes made after it lies
/// past one made before it, its slot where it takes one in the private pool. Each made and
/// released in an autorelease pool of its own, and Metal's side runs every shape first,
/// before this platform makes anything in the process: MoltenVK's images lie in the same
/// Metal space, where they would move Metal's next range
#[test]
fn depth_stencil_textures_without_optimized_contents_take_metals_space() {
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::rt::Pool;
    fn taken<P: Platform>(d: &P::Device, desc: &P::TextureDescriptor) -> (u64, u64) {
        let _pool = Pool::here();
        let before = next_range::<P>(d);
        let small = || {
            let buffer = d.new_buffer_of_length(OnFault::Die, 0x100, 0x20);
            buffer.unwrap().unwrap()
        };
        let first = small();
        let texture = d.new_texture(OnFault::Die, desc).unwrap().unwrap();
        let second = small();
        let range = next_range::<P>(d) - before;
        let slot = second.gpu_address() - first.gpu_address() - 0x100;
        drop((texture, first, second));
        (range, slot)
    }
    let mut shapes = Vec::new();
    for pixel_format in [260, 252, 250, 253] {
        for side in [4, 8, 16, 64, 100, 128] {
            for sample_count in [1, 4] {
                shapes.push(TextureDesc {
                    texture_type: if sample_count > 1 { 4 } else { 2 },
                    pixel_format,
                    width: side,
                    height: side,
                    sample_count,
                    usage: 0x5,
                    resource_options: Some(0x20),
                    allow_gpu_optimized_contents: Some(false),
                    ..TextureDesc::default()
                });
            }
        }
    }
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let theirs: Vec<_> = {
        let _queue = md.new_command_queue().unwrap();
        (shapes.iter())
            .map(|t| {
                let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(t);
                taken::<MetalPlatform>(&md, &desc)
            })
            .collect()
    };
    let (p, d) = (platform(), dev());
    let _queue = d.new_command_queue().unwrap();
    let differ: Vec<String> = (shapes.iter().zip(theirs))
        .filter_map(|(t, theirs)| {
            let ours = taken::<VulkanPlatform>(&d, &p.texture_descriptor(t));
            (ours != theirs).then(|| {
                let (format, side, samples) = (t.pixel_format, t.width, t.sample_count);
                format!("{format} {side}x{side} {samples}x: ours {ours:x?}, Metal's {theirs:x?}")
            })
        })
        .collect();
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A depth or stencil texture's heap size with allowGPUOptimizedContents off is larger at
/// the runtime than the Metal platform's on this host, as the runtime keeps its GPU-optimized
/// layout (`OptimizedDepthStencilTexture`), so the query answers Metal's, the runtime's model
/// of it or made of the runtime's answers for colour formats
/// (`metal_space::metal_size_and_align`): the Metal
/// platform's size and alignment for each format, 4x4 to 128x128, of 1 and 4 samples, at
/// each of [`DEPTH_USAGES`], where the runtime's own answer is at least as many bytes, at an
/// alignment Metal's divides. An X32_Stencil8 texture, whose answer is neither rule's, answers
/// the runtime's model of Metal's, the Metal platform's. With it on or unset the size is
/// Metal's: a `Depth32Float_Stencil8` one's is the runtime's, which models its layout and
/// announces nothing, Metal's at this shape
#[test]
fn heap_depth_stencil_sizes_with_optimized_contents_off() {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let d = dev();
    let shape_of = |pixel_format, side, sample_count, allow, usage| TextureDesc {
        texture_type: if sample_count > 1 { 4 } else { 2 },
        pixel_format,
        width: side,
        height: side,
        sample_count,
        usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: allow,
        ..TextureDesc::default()
    };
    let shape = |pixel_format, side, sample_count, allow| {
        shape_of(pixel_format, side, sample_count, allow, 0x5)
    };
    let metal = |t: &TextureDesc| {
        MetalPlatform::system_default_device().map(|metal| {
            let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(t);
            metal.heap_texture_size_and_align(&desc).unwrap()
        })
    };
    for (pixel_format, runtimes, metals) in [(252, 66048, 65536), (260, 98816, 81920)] {
        let t = shape(pixel_format, 128, 1, Some(false));
        let off = platform().texture_descriptor(&t);
        let raw = off.fields().descriptor(sel, Maker::SizeQuery);
        let answer = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
        assert_eq!(answer.size, runtimes, "{pixel_format}");
        let ours = d.heap_texture_size_and_align(&off).unwrap();
        assert_eq!(ours.size, metals, "{pixel_format}");
        if let Some(theirs) = metal(&t) {
            assert_eq!(theirs.size, metals, "{pixel_format}");
        }
        for allow in [None, Some(true)] {
            let t = shape(pixel_format, 128, 1, allow);
            let on = platform().texture_descriptor(&t);
            let theirs = metal(&t);
            let ours = d.heap_texture_size_and_align(&on);
            if pixel_format == 260 {
                let raw = on.fields().descriptor(sel, Maker::SizeQuery);
                let answer = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
                let unmeasured = rt::Degradation::UnmeasuredHeapTextureSize;
                assert!(!crate::answer::RELAYED.lock().contains(&unmeasured));
                let ours = ours.as_ref().unwrap();
                assert_eq!((ours.size, ours.align), (answer.size, answer.align));
            }
            if let Some(theirs) = theirs {
                assert_eq!(ours.unwrap().size, theirs.size, "{pixel_format} {allow:?}");
            }
        }
    }
    let mut differ = Vec::new();
    for pixel_format in [260, 252, 250, 253] {
        for side in [4, 8, 16, 64, 100, 128] {
            for samples in [1, 4] {
                for usage in DEPTH_USAGES {
                    let t = shape_of(pixel_format, side, samples, Some(false), usage);
                    let case = format!("{pixel_format} {side} {samples}x usage {usage:#x}");
                    let desc = platform().texture_descriptor(&t);
                    let ours = d.heap_texture_size_and_align(&desc).unwrap();
                    let raw = desc.fields().descriptor(sel, Maker::SizeQuery);
                    let runtimes = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
                    if runtimes.size < ours.size || !runtimes.align.is_multiple_of(ours.align) {
                        differ.push(format!("{case}: the runtime's {runtimes:x?} under"));
                    }
                    if let Some(theirs) = metal(&t)
                        && ours != theirs
                    {
                        differ.push(format!("{case}: {ours:x?} {theirs:x?}"));
                    }
                }
            }
        }
    }
    assert!(differ.is_empty(), "ours, Metal's: {differ:#?}");
    let t = shape(261, 16, 1, Some(false));
    let x32s8 = platform().texture_descriptor(&t);
    let ours = d.heap_texture_size_and_align(&x32s8).unwrap();
    let raw = x32s8.fields().descriptor(sel, Maker::SizeQuery);
    assert_eq!(
        Some(ours),
        d.metal_answer_at(sel, &raw),
        "X32_Stencil8: the model's"
    );
    if let Some(theirs) = metal(&t) {
        assert_eq!(ours, theirs, "X32_Stencil8");
    }
}

/// The size query of each of [`LOSSLESS_MIPMAPPED_SHAPES`], which answers the runtime's model of
/// Metal's on Metal ([`lossless_mipmapped_heap_texture_sizes_are_modelled`], on every OS), is
/// the Metal platform's on this host. Here, not there: it asks Metal. For the `R32Uint` one the
/// device's image takes 0x5680 bytes, where Metal answers 0x5800
#[test]
fn lossless_mipmapped_heap_texture_sizes_are_metals() {
    let (p, d) = (platform(), dev());
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    for (name, t) in lossless_mipmapped_shapes() {
        let ours = d
            .heap_texture_size_and_align(&p.texture_descriptor(&t))
            .unwrap();
        let theirs = md
            .heap_texture_size_and_align(&mp.texture_descriptor(&t))
            .unwrap();
        assert_eq!(ours, theirs, "{name}");
    }
}

/// The depth and stencil shapes the device tests below ask of both platforms: a name, texture
/// type, levels (0 for the full chain), slices and samples. Each is but a 2D or multisample one
/// of one slice and level ([`heap_depth_stencil_sizes_with_optimized_contents_off`]'s).
const DEPTH_SHAPES: [(&str, u64, u64, u64, u64); 10] = [
    ("two levels", 2, 2, 1, 1),
    ("every level", 2, 0, 1, 1),
    ("a 2D array of 2", 3, 1, 2, 1),
    ("a 2D array of 3, every level", 3, 0, 3, 1),
    ("a cube", 5, 1, 1, 1),
    ("a cube, every level", 5, 0, 1, 1),
    ("a cube array of 2", 6, 1, 2, 1),
    ("a cube array of 2, every level", 6, 0, 2, 1),
    ("a cube array of 3, two levels", 6, 2, 3, 1),
    ("a multisample array of 2", 8, 1, 2, 4),
];

/// The usages the depth and stencil device tests ask at: `ShaderRead`, `ShaderRead |
/// ShaderWrite`, `RenderTarget`, `ShaderRead | RenderTarget` and `ShaderRead | ShaderWrite |
/// PixelFormatView`.
const DEPTH_USAGES: [u64; 5] = [0x1, 0x3, 0x4, 0x5, 0x13];

/// A `Private` texture of `pixel_format` of a shape of [`DEPTH_SHAPES`], `side` by `side`, of
/// `usage`, with allowGPUOptimizedContents off.
fn depth_shape(
    pixel_format: u64,
    (texture_type, levels, slices, samples): (u64, u64, u64, u64),
    side: u64,
    usage: u64,
) -> TextureDesc {
    TextureDesc {
        texture_type,
        pixel_format,
        width: side,
        height: side,
        mipmap_level_count: match levels {
            0 => u64::from(side.ilog2()) + 1,
            n => n,
        },
        array_length: slices,
        sample_count: samples,
        usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    }
}

/// A shape of [`DEPTH_MORE_SHAPES`]: a name, texture type, extent, levels (0 for the full
/// chain), slices and samples.
type DepthShape = (&'static str, u64, (u64, u64), u64, u64, u64);

/// The representative shapes of the classes the wider measurements found Metal answering the
/// rules for ([`sides_and_levels`], [`arrays_and_samples`]) that the device test below asks
/// of both platforms beside [`DEPTH_SHAPES`]: a name, texture type, extent, levels (0 for the
/// full chain), slices and samples.
const DEPTH_MORE_SHAPES: [DepthShape; 14] = [
    ("1x1", 2, (1, 1), 1, 1, 1),
    ("1440x1080", 2, (1440, 1080), 1, 1, 1),
    ("3840x2160, every level", 2, (3840, 2160), 0, 1, 1),
    ("1000x1000, three levels", 2, (1000, 1000), 3, 1, 1),
    ("333x777, every level", 2, (333, 777), 0, 1, 1),
    (
        "a 2D array of 8, 129x64, every level",
        3,
        (129, 64),
        0,
        8,
        1,
    ),
    ("a 2D array of 2048, 4x4", 3, (4, 4), 1, 2048, 1),
    ("a cube of 200, every level", 5, (200, 200), 0, 1, 1),
    (
        "a cube array of 4, 129, three levels",
        6,
        (129, 129),
        3,
        4,
        1,
    ),
    ("a cube array of 341, 4", 6, (4, 4), 1, 341, 1),
    ("2 samples, 1440x1080", 4, (1440, 1080), 1, 1, 2),
    ("4 samples, 1920x1080", 4, (1920, 1080), 1, 1, 4),
    (
        "a multisample array of 5 at 2 samples, 100x37",
        8,
        (100, 37),
        1,
        5,
        2,
    ),
    (
        "a multisample array of 8 at 4 samples, 129x64",
        8,
        (129, 64),
        1,
        8,
        4,
    ),
];

/// The usages the device test below asks [`DEPTH_MORE_SHAPES`] at beside [`DEPTH_USAGES`],
/// of the sets the measurements found Metal answering the rules for: none (`Unknown`),
/// `ShaderWrite | RenderTarget`, `ShaderRead | RenderTarget | PixelFormatView`, `ShaderRead
/// | RenderTarget` with 0x10000, and a macOS guest's 0x1c01d; and sets holding neither
/// `ShaderRead` nor `RenderTarget` but another bit: `ShaderWrite`, `PixelFormatView`,
/// `ShaderWrite | PixelFormatView`, and 0x4000, 0x8000 and 0x10000 alone.
const DEPTH_MORE_USAGES: [u64; 11] = [
    0x0, 0x6, 0x15, 0x10005, 0x1c01d, 0x2, 0x10, 0x12, 0x4000, 0x8000, 0x10000,
];

/// `heapTextureSizeAndAlignWithDescriptor:` of a depth or stencil texture with
/// allowGPUOptimizedContents off, of `Depth16Unorm`, `Depth32Float`, `Stencil8` and
/// `Depth32Float_Stencil8`, at `Private` and `Shared` storage: of each shape of
/// [`DEPTH_SHAPES`], 4x4 to 128x128, at each of [`DEPTH_USAGES`]; of each of
/// [`DEPTH_MORE_SHAPES`] at those and [`DEPTH_MORE_USAGES`]; and of a 64x64 2D one at
/// `ShaderRead | RenderTarget` with the write-combined CPU cache mode, untracked hazard
/// tracking, `framebufferOnly` and protection options 0x1. The query answers each, the Metal
/// platform's answer ([`crate::gpu::metal_space::metal_size_and_align`]), a multisample one
/// at `Shared` storage among them, whose image the runtime makes in memory of its own; and the
/// runtime's own answer
/// is at least as many bytes, at an alignment the model's divides, which a heap's placement
/// of such a texture relies on (`HeapObject::new_texture`)
#[test]
fn heap_depth_stencil_shapes_with_optimized_contents_off_are_metals() {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let (p, d) = (platform(), dev());
    let mut rows = Vec::new();
    for (name, texture_type, levels, slices, samples) in DEPTH_SHAPES {
        for side in [4u64, 16, 64, 128] {
            for usage in DEPTH_USAGES {
                let shape = (texture_type, levels, slices, samples);
                let t = depth_shape(0, shape, side, usage);
                rows.push((format!("{name} {side} usage {usage:#x}"), t));
            }
        }
    }
    for (name, texture_type, extent, levels, slices, samples) in DEPTH_MORE_SHAPES {
        for usage in DEPTH_USAGES.into_iter().chain(DEPTH_MORE_USAGES) {
            let t = wide_row(texture_type, extent, levels, slices, samples, usage);
            rows.push((format!("{name} usage {usage:#x}"), t));
        }
    }
    let plain_64 = wide_row(2, (64, 64), 1, 1, 1, 0x5);
    let properties: [(&str, TextureDesc); 4] = [
        (
            "write-combined",
            TextureDesc {
                resource_options: Some(0x1),
                ..plain_64
            },
        ),
        (
            "untracked",
            TextureDesc {
                resource_options: Some(0x100),
                ..plain_64
            },
        ),
        (
            "framebufferOnly",
            TextureDesc {
                framebuffer_only: Some(true),
                ..plain_64
            },
        ),
        (
            "protection options 0x1",
            TextureDesc {
                protection_options: Some(0x1),
                ..plain_64
            },
        ),
    ];
    for (name, t) in properties {
        rows.push((format!("64x64 {name}"), t));
    }
    let mut differ = Vec::new();
    for pixel_format in [250, 252, 253, 260] {
        for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
            for (name, row) in &rows {
                let case = format!("{pixel_format} {storage} {name}");
                let mut t = *row;
                t.pixel_format = pixel_format;
                // The row's options but its storage mode, which is the loop's.
                let others = row.resource_options.unwrap_or(0) & !0xf0;
                t.resource_options = Some(options | others);
                let ours = p.texture_descriptor(&t);
                *crate::answer::LAST_LINE.lock() = None;
                let answered = catch_death(|| d.heap_texture_size_and_align(&ours));
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let raw = ours.fields().descriptor(sel, Maker::SizeQuery);
                let answer = |desc: &rt::TextureDescriptor| d.size_and_align_at(sel, desc);
                let modelled = crate::gpu::metal_space::metal_size_and_align(&raw, answer);
                let runtimes = crate::answer::plain(|| d.raw.heap_texture_size_and_align(&raw));
                if let (Some(model), Ok(runtimes)) = (modelled, &runtimes)
                    && (runtimes.size < model.size || !runtimes.align.is_multiple_of(model.align))
                {
                    differ.push(format!("{case}: the runtime's {runtimes:x?} under"));
                }
                match answered {
                    Ok(ours) => {
                        let ours = ours.unwrap();
                        let theirs = MetalPlatform::system_default_device().map(|metal| {
                            let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&t);
                            metal.heap_texture_size_and_align(&desc).unwrap()
                        });
                        if theirs.is_some_and(|theirs| theirs != ours) {
                            differ.push(format!("{case}: ours {ours:x?}, Metal's {theirs:x?}"));
                        }
                    }
                    Err(site) => differ.push(format!("{case}: died at {site:?}: {line}")),
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A measurement: Metal's `heapTextureSizeAndAlignWithDescriptor:` of a depth or stencil
/// texture with allowGPUOptimizedContents off, of `Depth16Unorm`, `Depth32Float`, `Stencil8`,
/// `Depth32Float_Stencil8` and `X32_Stencil8`, at `Private` and `Shared` storage, of one
/// image (2D and multisample) and of each shape of [`DEPTH_SHAPES`], 4x4 to 128x128, at each
/// of [`DEPTH_USAGES`], against the rules [`crate::gpu::metal_space::metal_size_and_align`]
/// makes Metal's answer by: a format of one aspect as its colour format, the runtime's answer
/// for it, and a two-aspect one as its depth's answer rounded up to a page, then its
/// stencil's ([`crate::gpu::metal_space::paired`]). It prints each format, storage and shape,
/// whether every row of it is the rule's, and each row Metal answers otherwise, Metal's answer
/// beside the rule's. Every row of the formats the rules are taken for is the rule's
/// (`twins`); `X32_Stencil8`'s, which no rule is taken for, are printed alone
#[test]
#[ignore = "a measurement: Metal's depth and stencil sizes against the rules"]
fn depth_stencil_sizes_against_the_rules() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let d = dev();
    let one_image = [
        ("one image", 2, 1, 1, 1),
        ("one 4-sample image", 4, 1, 1, 4),
    ];
    let mut differ = Vec::new();
    for pixel_format in [250, 252, 253, 260, 261] {
        for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
            for (name, texture_type, levels, slices, samples) in
                one_image.into_iter().chain(DEPTH_SHAPES)
            {
                let mut rows = Vec::new();
                for side in [4u64, 16, 64, 128] {
                    for usage in DEPTH_USAGES {
                        let shape = (texture_type, levels, slices, samples);
                        let mut t = depth_shape(pixel_format, shape, side, usage);
                        t.resource_options = Some(options);
                        let raw = platform()
                            .texture_descriptor(&t)
                            .fields()
                            .descriptor(sel, Maker::SizeQuery);
                        let answer = |desc: &rt::TextureDescriptor| d.size_and_align_at(sel, desc);
                        let colour = match pixel_format {
                            250 => Some(PixelFormat::R16Unorm),
                            252 => Some(PixelFormat::R32Float),
                            253 => Some(PixelFormat::R8Unorm),
                            _ => None,
                        };
                        let rule = match colour {
                            Some(colour) => {
                                let mut twin = raw;
                                twin.pixel_format = colour;
                                answer(&twin)
                            }
                            None => crate::gpu::metal_space::paired(&raw, answer),
                        };
                        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&t);
                        let theirs = metal.heap_texture_size_and_align(&desc).unwrap();
                        if rule != Some(theirs) {
                            rows.push(format!(
                                "  {side} usage {usage:#x}: Metal's {theirs:x?}, the rule's \
                                 {rule:x?}"
                            ));
                        }
                    }
                }
                let verdict = if rows.is_empty() {
                    "the rule's"
                } else {
                    "OTHER"
                };
                eprintln!("{pixel_format} {storage} {name}: {verdict}");
                for row in &rows {
                    eprintln!("{row}");
                }
                if !rows.is_empty() && pixel_format != 261 {
                    differ.push(format!("{pixel_format} {storage} {name}"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The rules' answer for a depth or stencil texture of `raw`, of `d`'s answers for colour
/// formats ([`crate::gpu::metal_space::metal_size_and_align`]'s, where it takes them): a
/// format of one aspect as its colour format, and any other as its depth's answer rounded up
/// to a page, then its stencil's ([`crate::gpu::metal_space::paired`]). `None` where the
/// runtime answers none.
fn depth_rule(d: &VulkanDevice, raw: &rt::TextureDescriptor) -> Option<SizeAndAlign> {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let answer = |desc: &rt::TextureDescriptor| d.size_and_align_at(sel, desc);
    let colour = match raw.pixel_format {
        PixelFormat::Depth16Unorm => PixelFormat::R16Unorm,
        PixelFormat::Depth32Float => PixelFormat::R32Float,
        PixelFormat::Stencil8 => PixelFormat::R8Unorm,
        _ => return crate::gpu::metal_space::paired(raw, answer),
    };
    let mut twin = *raw;
    twin.pixel_format = colour;
    answer(&twin)
}

/// The square sides the wider depth and stencil measurement asks ([`measure_depth_classes`]):
/// 1 to 16384, odd ones and ones a texel past a power of two among them.
const WIDE_SIDES: [u64; 32] = [
    1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256, 257,
    511, 512, 513, 1000, 1023, 1024, 2048, 4096, 16384,
];

/// The extents of other widths and heights the wider measurement asks: displays' and odd
/// ones, and one texel across.
const WIDE_EXTENTS: [(u64, u64); 16] = [
    (1440, 1080),
    (3840, 2160),
    (1920, 1080),
    (2560, 1600),
    (5120, 2880),
    (1024, 768),
    (3, 5),
    (7, 3),
    (100, 37),
    (129, 64),
    (64, 129),
    (333, 777),
    (1, 1000),
    (1000, 1),
    (16384, 1),
    (1, 16384),
];

/// A few extents of each kind, for the shapes whose rows multiply (long arrays).
const WIDE_FEW: [(u64, u64); 6] = [
    (4, 4),
    (64, 64),
    (129, 129),
    (1000, 1000),
    (1440, 1080),
    (3, 5),
];

/// The usage bits the wider measurement's usage classes take sets of: `ShaderRead`,
/// `ShaderWrite`, `RenderTarget`, 0x8 (which Metal drops), `PixelFormatView`, and Apple's
/// private 0x4000, 0x8000 and 0x10000: every bit Metal defines but `ShaderAtomic` (0x20),
/// which Metal asserts at for every depth and stencil format ("is not a valid format to be
/// used with texture atomics").
const WIDE_USAGE_BITS: [u64; 8] = [0x1, 0x2, 0x4, 0x8, 0x10, 0x4000, 0x8000, 0x10000];

/// Every set of [`WIDE_USAGE_BITS`], none (`Unknown`) among them, but [`DEPTH_USAGES`]'.
fn wide_usages() -> Vec<u64> {
    (0..1u64 << WIDE_USAGE_BITS.len())
        .map(|set| {
            (WIDE_USAGE_BITS.iter().enumerate())
                .filter(|&(i, _)| set & (1 << i) != 0)
                .fold(0, |usage, (_, bit)| usage | bit)
        })
        .filter(|usage| !DEPTH_USAGES.contains(usage))
        .collect()
}

/// The representative shapes the wider measurement asks its usages, options and private
/// properties of, one or two of each of its shape classes: 2D of one level (64x64, 1440x1080,
/// 1x1), 2D of every level (200x100, 3840x2160) and of three (1000x1000), 2D arrays of 3 of
/// two levels (129x129) and of 2048 of every level (4x4), a cube of every level (128), cube
/// arrays of 2 of every level (64) and of 341 of one (4), 2D multisample at 2 samples
/// (1440x1080) and 4 (100x37), and multisample arrays of 3 at 2 samples (64x64) and of 8 at 4
/// (129x64), of `usage`.
fn wide_representatives(usage: u64) -> [TextureDesc; 15] {
    [
        wide_row(2, (64, 64), 1, 1, 1, usage),
        wide_row(2, (1440, 1080), 1, 1, 1, usage),
        wide_row(2, (1, 1), 1, 1, 1, usage),
        wide_row(2, (200, 100), 0, 1, 1, usage),
        wide_row(2, (3840, 2160), 0, 1, 1, usage),
        wide_row(2, (1000, 1000), 3, 1, 1, usage),
        wide_row(3, (129, 129), 2, 3, 1, usage),
        wide_row(3, (4, 4), 0, 2048, 1, usage),
        wide_row(5, (128, 128), 0, 1, 1, usage),
        wide_row(6, (64, 64), 0, 2, 1, usage),
        wide_row(6, (4, 4), 1, 341, 1, usage),
        wide_row(4, (1440, 1080), 1, 1, 2, usage),
        wide_row(4, (100, 37), 1, 1, 4, usage),
        wide_row(8, (64, 64), 1, 3, 2, usage),
        wide_row(8, (129, 64), 1, 8, 4, usage),
    ]
}

/// A class of the wider measurement: its name, whether Metal answers the rules' size at
/// every row of it, and its rows (of no pixel format or storage yet).
type DepthClass = (&'static str, bool, Vec<TextureDesc>);

/// The rows `f` makes of each extent of `of`, at each of [`DEPTH_USAGES`].
fn at_each_usage(
    of: &[(u64, u64)],
    f: impl Fn(u64, (u64, u64)) -> Vec<TextureDesc>,
) -> Vec<TextureDesc> {
    (DEPTH_USAGES.iter())
        .flat_map(|&usage| of.iter().flat_map(|&e| f(usage, e)).collect::<Vec<_>>())
        .collect()
}

/// Every extent of the wider measurement: [`WIDE_SIDES`]' squares, then [`WIDE_EXTENTS`].
fn wide_extents() -> Vec<(u64, u64)> {
    (WIDE_SIDES.iter().map(|&s| (s, s)))
        .chain(WIDE_EXTENTS)
        .collect()
}

/// The square extents of [`WIDE_SIDES`].
fn wide_squares() -> Vec<(u64, u64)> {
    WIDE_SIDES.iter().map(|&s| (s, s)).collect()
}

/// The wider measurement's classes of sides and levels: 2D of one level and of every level
/// count, at every extent, and cubes and cube arrays of 1 to 4 of every level count, at
/// every square side. Metal answers the rules' at each.
fn sides_and_levels() -> Vec<DepthClass> {
    let (extents, squares) = (wide_extents(), wide_squares());
    vec![
        (
            "2D of one level, sides 1 to 16384 and others",
            true,
            at_each_usage(&extents, |u, e| vec![wide_row(2, e, 1, 1, 1, u)]),
        ),
        (
            "2D of every level count",
            true,
            at_each_usage(&extents, |u, (w, h)| {
                (2..=full_chain(w, h))
                    .map(|levels| wide_row(2, (w, h), levels, 1, 1, u))
                    .collect()
            }),
        ),
        (
            "cubes of every level count",
            true,
            at_each_usage(&squares, |u, (w, h)| {
                (1..=full_chain(w, h))
                    .map(|levels| wide_row(5, (w, h), levels, 1, 1, u))
                    .collect()
            }),
        ),
        (
            "cube arrays of 1 to 4, of every level count",
            true,
            at_each_usage(&squares, |u, (w, h)| {
                (1..=4)
                    .flat_map(|cubes| {
                        (1..=full_chain(w, h)).map(move |l| wide_row(6, (w, h), l, cubes, 1, u))
                    })
                    .collect()
            }),
        ),
    ]
}

/// The wider measurement's classes of arrays and sample counts: 2D arrays of 2 to 8 of one,
/// two and every level at every extent, of 16, 64 and 2048 at a few, and of every length to
/// 2048 at one; cube arrays of 8 and 341 and of every length to 341; 2D multisample at 2
/// and 4 samples at every extent; multisample arrays of 1 to 8 at every extent, of 16, 64 and
/// 2048 at a few, and of every length to 2048 at one. Metal answers the rules' at each.
fn arrays_and_samples() -> Vec<DepthClass> {
    let extents = wide_extents();
    let every_length = |texture_type, lengths: std::ops::RangeInclusive<u64>, samples| {
        lengths
            .flat_map(|n| match samples {
                1 => [1, 0]
                    .map(|l| wide_row(texture_type, (129, 129), l, n, 1, 0x5))
                    .to_vec(),
                _ => vec![wide_row(texture_type, (129, 64), 1, n, samples, 0x5)],
            })
            .collect::<Vec<_>>()
    };
    vec![
        (
            "2D arrays of 2 to 8, of one, two and every level",
            true,
            at_each_usage(&extents, |u, e| {
                (2..=8)
                    .flat_map(|slices| [1, 2, 0].map(|l| wide_row(3, e, l, slices, 1, u)))
                    .filter(|t| t.mipmap_level_count <= full_chain(t.width, t.height))
                    .collect()
            }),
        ),
        (
            "2D arrays of 16, 64 and 2048, of one and every level",
            true,
            at_each_usage(&WIDE_FEW, |u, e| {
                [16, 64, 2048]
                    .into_iter()
                    .flat_map(|slices| [1, 0].map(|l| wide_row(3, e, l, slices, 1, u)))
                    .collect()
            }),
        ),
        (
            "2D arrays of every length to 2048, 129x129, of one and every level",
            true,
            every_length(3, 1..=2048, 1),
        ),
        (
            "cube arrays of 8 and 341, of one and every level",
            true,
            at_each_usage(&[(4, 4), (64, 64), (129, 129), (1000, 1000)], |u, e| {
                [8, 341]
                    .into_iter()
                    .flat_map(|cubes| [1, 0].map(|l| wide_row(6, e, l, cubes, 1, u)))
                    .collect()
            }),
        ),
        (
            "cube arrays of every length to 341, 129, of one and every level",
            true,
            every_length(6, 1..=341, 1),
        ),
        (
            "2D multisample at 2 samples",
            true,
            at_each_usage(&extents, |u, e| vec![wide_row(4, e, 1, 1, 2, u)]),
        ),
        (
            "2D multisample at 4 samples",
            true,
            at_each_usage(&extents, |u, e| vec![wide_row(4, e, 1, 1, 4, u)]),
        ),
        (
            "multisample arrays of 1 to 8 at 2 and 4 samples",
            true,
            at_each_usage(&extents, |u, e| {
                (1..=8)
                    .flat_map(|s| [2, 4].map(|n| wide_row(8, e, 1, s, n, u)))
                    .collect()
            }),
        ),
        (
            "multisample arrays of 16, 64 and 2048 at 2 and 4 samples",
            true,
            at_each_usage(&WIDE_FEW, |u, e| {
                [16, 64, 2048]
                    .into_iter()
                    .flat_map(|s| [2, 4].map(|n| wide_row(8, e, 1, s, n, u)))
                    .collect()
            }),
        ),
        (
            "multisample arrays of every length to 2048, 129x64, at 2 and 4 samples",
            true,
            (every_length(8, 1..=2048, 2).into_iter())
                .chain(every_length(8, 1..=2048, 4))
                .collect(),
        ),
    ]
}

/// Whether `usage`, a set of [`WIDE_USAGE_BITS`], holds `ShaderRead` or `RenderTarget`, or no
/// bit but 0x8, which Metal drops: the sets Metal was first measured to answer the rules' for at
/// every row, which the others follow too.
fn read_or_rendered(usage: u64) -> bool {
    usage & !0x8 == 0 || usage & 0x5 != 0
}

/// The wider measurement's classes of usages, options and private properties, each of the
/// representative shapes ([`wide_representatives`]): every set of [`WIDE_USAGE_BITS`] but
/// [`DEPTH_USAGES`], those holding `ShaderRead` or `RenderTarget` or no bit but 0x8, and the
/// others, each of which Metal answers the rules' for (the others:
/// [`crate::gpu::metal_space`]'s `twins`); and at
/// each of [`DEPTH_USAGES`], the write-combined CPU cache mode, untracked and tracked hazard
/// tracking, `framebufferOnly`, `isDrawable`, protection options 0x1, 0x2 and bit 63, and
/// `writeSwizzleEnabled` (of the identity swizzle), and a swizzle at `ShaderRead`, which it
/// answers the rules' for.
fn usages_and_properties() -> Vec<DepthClass> {
    let usages = |read_or_rendered_only| {
        (wide_usages().into_iter())
            .filter(|&usage| read_or_rendered(usage) == read_or_rendered_only)
            .flat_map(wide_representatives)
            .collect()
    };
    let with = |f: &dyn Fn(&mut TextureDesc)| {
        (DEPTH_USAGES.iter())
            .flat_map(|&usage| wide_representatives(usage))
            .map(|mut t| {
                f(&mut t);
                t
            })
            .collect::<Vec<_>>()
    };
    vec![
        (
            "usage sets holding ShaderRead or RenderTarget, or no bit but 0x8",
            true,
            usages(true),
        ),
        (
            "usage sets holding neither ShaderRead nor RenderTarget, but another bit",
            true,
            usages(false),
        ),
        (
            "CPU cache mode write-combined",
            true,
            with(&|t| t.resource_options = Some(0x1)),
        ),
        (
            "hazard tracking untracked",
            true,
            with(&|t| t.resource_options = Some(0x100)),
        ),
        (
            "hazard tracking tracked",
            true,
            with(&|t| t.resource_options = Some(0x200)),
        ),
        (
            "framebufferOnly",
            true,
            with(&|t| t.framebuffer_only = Some(true)),
        ),
        ("isDrawable", true, with(&|t| t.is_drawable = Some(true))),
        (
            "protection options 0x1, 0x2 and bit 63",
            true,
            [0x1, 0x2, 1 << 63]
                .into_iter()
                .flat_map(|p| with(&move |t| t.protection_options = Some(p)))
                .collect(),
        ),
        (
            "writeSwizzleEnabled, of the identity swizzle",
            true,
            with(&|t| t.write_swizzle_enabled = Some(true)),
        ),
        (
            "a swizzle (alpha, blue, green, red), read alone",
            true,
            wide_representatives(0x1)
                .into_iter()
                .map(|mut t| {
                    t.swizzle = Some(Swizzle {
                        red: 5,
                        green: 4,
                        blue: 3,
                        alpha: 2,
                    });
                    t
                })
                .collect(),
        ),
    ]
}

/// Asks Metal's `heapTextureSizeAndAlignWithDescriptor:` of each row of `classes` with
/// allowGPUOptimizedContents off, of `Depth16Unorm`, `Depth32Float`, `Stencil8` and
/// `Depth32Float_Stencil8`, at `Private` and `Shared` storage, against the rules
/// ([`depth_rule`]), and prints, for each class, format and storage, whether every row of
/// it is the rule's, up to 12 rows Metal answers otherwise (Metal's answer beside the
/// rule's), the rows the runtime answers none for (its colour answer refused), and the rows
/// whose runtime answer is under the rule's (where a heap's placement dies,
/// `HeapObject::optimized_made`). Every row of a class marked so is the rule's, and some row
/// of each other class is not.
fn measure_depth_classes(classes: Vec<DepthClass>) {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let (p, d) = (platform(), dev());
    let mut unexpected = Vec::new();
    for (name, the_rules, rows) in classes {
        for pixel_format in [250, 252, 253, 260] {
            for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
                let (mut other, mut none, mut under) = (Vec::new(), 0, 0);
                for row in &rows {
                    let mut t = *row;
                    t.pixel_format = pixel_format;
                    t.resource_options = Some(options | row.resource_options.unwrap_or(0));
                    let raw = (p.texture_descriptor(&t).fields()).descriptor(sel, Maker::SizeQuery);
                    let rule = depth_rule(&d, &raw);
                    let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&t);
                    let theirs = metal.heap_texture_size_and_align(&desc).unwrap();
                    let shape = format!(
                        "type {} {}x{} levels {} slices {} samples {} usage {:#x}",
                        t.texture_type,
                        t.width,
                        t.height,
                        t.mipmap_level_count,
                        t.array_length,
                        t.sample_count,
                        t.usage,
                    );
                    let Some(rule) = rule else {
                        none += 1;
                        other.push(format!("  {shape}: Metal's {theirs:x?}, the rule's none"));
                        continue;
                    };
                    if rule != theirs {
                        other.push(format!(
                            "  {shape}: Metal's {theirs:x?}, the rule's {rule:x?}"
                        ));
                    }
                    let runtimes = crate::answer::plain(|| d.raw.heap_texture_size_and_align(&raw));
                    if runtimes
                        .is_ok_and(|r| r.size < rule.size || !r.align.is_multiple_of(rule.align))
                    {
                        under += 1;
                    }
                }
                let verdict = match other.len() {
                    0 => "the rule's".to_owned(),
                    n => format!("OTHER at {n}"),
                };
                eprintln!(
                    "{name}: {pixel_format} {storage}: {verdict} of {} rows; the runtime \
                     answers none at {none}; its own answer under the rule's at {under}",
                    rows.len()
                );
                for row in other.iter().take(12) {
                    eprintln!("{row}");
                }
                if other.is_empty() != the_rules {
                    unexpected.push(format!("{name}: {pixel_format} {storage}"));
                }
            }
        }
    }
    assert!(unexpected.is_empty(), "{unexpected:#?}");
}

/// A measurement ([`measure_depth_classes`]) of the wider measurement's sides and levels
/// ([`sides_and_levels`]), against the rules
#[test]
#[ignore = "a measurement: Metal's depth and stencil sizes of more sides and levels"]
fn depth_stencil_sizes_of_more_sides_and_levels_against_the_rules() {
    measure_depth_classes(sides_and_levels());
}

/// A measurement ([`measure_depth_classes`]) of the wider measurement's arrays and sample
/// counts ([`arrays_and_samples`]), against the rules
#[test]
#[ignore = "a measurement: Metal's depth and stencil sizes of more arrays and samples"]
fn depth_stencil_sizes_of_more_arrays_and_samples_against_the_rules() {
    measure_depth_classes(arrays_and_samples());
}

/// A measurement ([`measure_depth_classes`]) of the wider measurement's usages, options and
/// private properties ([`usages_and_properties`]), against the rules
#[test]
#[ignore = "a measurement: Metal's depth and stencil sizes of more usages and properties"]
fn depth_stencil_sizes_of_more_usages_and_properties_against_the_rules() {
    measure_depth_classes(usages_and_properties());
}

/// Asks Metal's `heapTextureSizeAndAlignWithDescriptor:` of an `X32_Stencil8` texture of each
/// row of the wider measurement's classes ([`sides_and_levels`], [`arrays_and_samples`],
/// [`usages_and_properties`]), with allowGPUOptimizedContents off, at `Private` and `Shared`
/// storage, against the runtime's model of Metal's answer (`VulkanDevice::metal_answer_at`,
/// which sizes it as a `Stencil8` texture of its shape), the size such a texture's range of
/// Metal's space takes ([`crate::gpu::metal_space::own_heap_size`]). Prints, for each class and
/// storage, how many rows the model answers as Metal, how many otherwise and how many it
/// answers none for (where `newTextureWithDescriptor:` dies), with up to 12 rows of each; and
/// asserts that the model answers every row, each as Metal does.
#[test]
#[ignore = "a measurement: Metal's X32_Stencil8 sizes against the runtime's model"]
fn x32_stencil8_sizes_against_the_model() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let (p, d) = (platform(), dev());
    let mut unexpected = Vec::new();
    let classes = (sides_and_levels().into_iter())
        .chain(arrays_and_samples())
        .chain(usages_and_properties());
    for (name, _, rows) in classes {
        for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
            let (mut other, mut none, mut equal) = (Vec::new(), Vec::new(), 0);
            for row in &rows {
                let mut t = *row;
                t.pixel_format = 261;
                t.resource_options = Some(options | row.resource_options.unwrap_or(0));
                let raw = (p.texture_descriptor(&t).fields()).descriptor(sel, Maker::SizeQuery);
                let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&t);
                let theirs = metal.heap_texture_size_and_align(&desc).unwrap();
                let shape = format!(
                    "type {} {}x{} levels {} slices {} samples {} usage {:#x}",
                    t.texture_type,
                    t.width,
                    t.height,
                    t.mipmap_level_count,
                    t.array_length,
                    t.sample_count,
                    t.usage,
                );
                match d.metal_answer_at(sel, &raw) {
                    Some(model) if model == theirs => equal += 1,
                    Some(model) => other.push(format!(
                        "  {shape}: Metal's {theirs:x?}, the model's {model:x?}"
                    )),
                    None => none.push(format!("  {shape}: Metal's {theirs:x?}, the model's none")),
                }
            }
            eprintln!(
                "{name}: X32_Stencil8 {storage}: the model's at {equal} of {} rows, other at {}, \
                 none at {}",
                rows.len(),
                other.len(),
                none.len()
            );
            for row in other.iter().take(12).chain(none.iter().take(12)) {
                eprintln!("{row}");
            }
            if !(other.is_empty() && none.is_empty()) {
                unexpected.push(format!("{name}: {storage}"));
            }
        }
    }
    assert!(unexpected.is_empty(), "{unexpected:#?}");
}

/// `newTextureWithDescriptor:` of a `Depth32Float_Stencil8` or `X32_Stencil8` texture with
/// allowGPUOptimizedContents off, whose image the runtime keeps GPU-optimized
/// (`optimized-depth-stencil-texture`): each is made, the suite's (a 2D one of two levels and a
/// cube array of two cubes of one level, 4x4 at `Private` storage, of usage `ShaderRead |
/// ShaderWrite`, and an `X32_Stencil8` one, 2D, of one level) among them, and its range of
/// Metal's space is of Metal's size as the runtime models it
/// ([`crate::gpu::metal_space::own_heap_size`]), the Metal platform's
/// answer for each here, as it is at every shape `x32_stencil8_sizes_against_the_model`
/// measures. A texture the model answers none for dies at the selector, naming the kind. The
/// Metal platform makes each
#[test]
fn optimized_depth_textures_take_metals_modelled_size() {
    let sel = "newTextureWithDescriptor:";
    let sized = |side, usage, pixel_format, texture_type, levels, slices| TextureDesc {
        texture_type,
        pixel_format,
        width: side,
        height: side,
        mipmap_level_count: levels,
        array_length: slices,
        usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let shape = |pixel_format, texture_type, levels, slices| {
        sized(4, 0x3, pixel_format, texture_type, levels, slices)
    };
    let shapes = [
        ("2D, two levels", shape(260, 2, 2, 1)),
        ("a cube array, one level", shape(260, 6, 1, 2)),
        ("a 2D array", shape(260, 3, 1, 2)),
        ("a cube", shape(260, 5, 1, 1)),
        ("a cube array, two levels", shape(260, 6, 2, 2)),
        ("a cube array of three", shape(260, 6, 1, 3)),
        ("2D, two levels, 8x8", sized(8, 0x3, 260, 2, 2, 1)),
        (
            "2D, two levels, read and render",
            sized(4, 0x5, 260, 2, 2, 1),
        ),
        ("X32_Stencil8 2D, one level", shape(261, 2, 1, 1)),
        ("X32_Stencil8 2D, two levels", shape(261, 2, 2, 1)),
        ("an X32_Stencil8 2D array", shape(261, 3, 1, 2)),
        (
            "X32_Stencil8 2D, one level, 8x8",
            sized(8, 0x3, 261, 2, 1, 1),
        ),
        (
            "X32_Stencil8 2D, one level, read and render",
            sized(4, 0x5, 261, 2, 1, 1),
        ),
    ];
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let d = dev();
    for (name, t) in &shapes {
        let desc = platform().texture_descriptor(t);
        let made = d.new_texture(OnFault::Return, &desc);
        assert!(made.unwrap().is_some(), "{name}");
        let raw = desc.fields().descriptor(sel, Maker::Other);
        let modelled = d.metal_answer_at(sel, &raw);
        if let Some(md) = &md {
            let theirs = md.new_texture(OnFault::Return, &mp.texture_descriptor(t));
            assert!(theirs.unwrap().is_some(), "Metal: {name}");
            let size = md.heap_texture_size_and_align(&mp.texture_descriptor(t));
            assert_eq!(modelled, Some(size.unwrap()), "Metal's size: {name}");
        }
    }
}

/// `heapTextureSizeAndAlignWithDescriptor:` of an `X32_Stencil8` texture with
/// allowGPUOptimizedContents off, whose image the runtime keeps GPU-optimized
/// (`optimized-depth-stencil-texture`), answers the runtime's model of Metal's answer, a
/// `Stencil8` texture of its shape, the size `newTextureWithDescriptor:` gives its range of
/// Metal's space: the Metal platform's size and alignment for each shape here, at `Private` and
/// `Shared` storage, a 2D one of two levels among them. It died at its selector before, as the
/// colour rule answers none for it (`metal_space::metal_size_and_align`)
#[test]
fn x32_stencil8_size_queries_are_metals() {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let sized = |side, usage, texture_type, levels, slices, options| TextureDesc {
        texture_type,
        pixel_format: 261,
        width: side,
        height: side,
        mipmap_level_count: levels,
        array_length: slices,
        usage,
        resource_options: Some(options),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let mut shapes = Vec::new();
    for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
        for (name, t) in [
            ("2D, one level", sized(4, 0x3, 2, 1, 1, options)),
            ("2D, two levels", sized(4, 0x3, 2, 2, 1, options)),
            ("a 2D array", sized(4, 0x3, 3, 1, 2, options)),
            ("a cube", sized(4, 0x3, 5, 1, 1, options)),
            ("2D, one level, 8x8", sized(8, 0x3, 2, 1, 1, options)),
            ("2D, two levels, 64x64", sized(64, 0x1, 2, 2, 1, options)),
            (
                "2D, one level, read and render",
                sized(4, 0x5, 2, 1, 1, options),
            ),
        ] {
            shapes.push((format!("{storage} {name}"), t));
        }
    }
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let theirs: Vec<_> = (shapes.iter())
        .map(|(_, t)| {
            (md.as_ref()).map(|md| {
                md.heap_texture_size_and_align(&mp.texture_descriptor(t))
                    .unwrap()
            })
        })
        .collect();
    let d = dev();
    for ((name, t), theirs) in shapes.iter().zip(theirs) {
        let desc = platform().texture_descriptor(t);
        let ours = d.heap_texture_size_and_align(&desc).unwrap();
        let raw = desc.fields().descriptor(sel, Maker::SizeQuery);
        assert_eq!(
            Some(ours),
            d.metal_answer_at(sel, &raw),
            "the model's: {name}"
        );
        if let Some(theirs) = theirs {
            assert_eq!(ours, theirs, "Metal's: {name}");
        }
        let made = d.new_texture(OnFault::Return, &desc);
        assert!(made.unwrap().is_some(), "made: {name}");
    }
}

/// A texture or view of `BGRA10_XR`, `BGRA10_XR_sRGB`, `BGR10_XR` or `BGR10_XR_sRGB` under a
/// swizzle other than the identity is made, as Metal makes it: a device's texture, a heap's
/// placement, and a view of an unswizzled texture by either swizzle selector. The runtime
/// decodes such a texture's codes before it swizzles them, a border texel included, so its
/// reads, samples and gathers are Metal's (`rgbx`'s
/// `extended_range_textures_read_as_metals` compares them)
#[test]
fn swizzled_extended_range_textures_are_made() {
    let md = MetalPlatform::system_default_device();
    let (p, d) = (platform(), dev());
    // Alpha, red, one, zero.
    let swizzle = Swizzle {
        red: 5,
        green: 2,
        blue: 1,
        alpha: 0,
    };
    let desc = |pixel_format, swizzle| TextureDesc {
        pixel_format,
        width: 16,
        height: 16,
        usage: 1,
        resource_options: Some(0x20),
        swizzle,
        ..TextureDesc::default()
    };
    // A placement heap of 1 MiB of `Private` storage.
    fn heap_desc<'a, P: Platform>() -> HeapDesc<'a, P> {
        HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        }
    }
    let whole = NsRange::new(0, 1);
    let formats = [
        (552, "BGRA10_XR"),
        (553, "BGRA10_XR_sRGB"),
        (554, "BGR10_XR"),
        (555, "BGR10_XR_sRGB"),
    ];
    for (format, name) in formats {
        if let Some(md) = &md {
            let metal = |s| reims_vgpu_metal::gpu::desc::texture_descriptor(&desc(format, s));
            let made = md.new_texture(OnFault::Return, &metal(Some(swizzle)));
            assert!(made.unwrap().is_some(), "{name}: Metal's texture");
            let heap = md.new_heap(&heap_desc()).unwrap().heap.unwrap();
            let placed = heap.new_texture(&metal(Some(swizzle)), 0).unwrap();
            assert!(placed.is_some(), "{name}: Metal's placement");
            let plain = md.new_texture(OnFault::Return, &metal(None));
            let plain = plain.unwrap().unwrap();
            let view = plain.new_swizzled_texture_subview(format, 2, whole, whole, swizzle);
            assert!(view.unwrap().is_some(), "{name}: Metal's view");
        }
        let ours = |s| p.texture_descriptor(&desc(format, s));
        let made = d.new_texture(OnFault::Return, &ours(Some(swizzle)));
        assert!(made.unwrap().is_some(), "{name}: texture");
        let heap = d.new_heap(&heap_desc()).unwrap().heap.unwrap();
        let placed = heap.new_texture(&ours(Some(swizzle)), 0).unwrap();
        assert!(placed.is_some(), "{name}: placement");
        let plain = d
            .new_texture(OnFault::Return, &ours(None))
            .unwrap()
            .unwrap();
        let view = plain.new_swizzled_texture_subview(format, 2, whole, whole, swizzle);
        assert!(view.unwrap().is_some(), "{name}: view");
        let view =
            plain.new_swizzled_texture_subview_at_index(format, 2, whole, whole, swizzle, 0x800);
        assert!(view.unwrap().is_some(), "{name}: view at an index");
    }
}

/// A swizzle other than the identity over memory whose GPU writes never reach it (a surface
/// of `kIOSurfaceClientDirection` 2 over a guest's page, and a buffer made on it) is refused
/// under `ShaderWrite` or `RenderTarget` usage as over any other memory, Metal's answers on
/// the M3 Max under macOS 26.7, one case a process: without `writeSwizzleEnabled` Metal's
/// validation asserts ("Texture Descriptor Validation"), where the runtime's validation stops
/// it (`SwizzleWithWrites`); with it Metal makes the texture, of the usage asked, and writes
/// through the swizzle, which the runtime does not build, so this platform dies naming the
/// gap (`RuntimeGap::WriteSwizzle`) in strict mode, as this crate's tests run, and answers nil
/// by default (ours, `crate::answer::nil_for_gap`). Read alone (`ShaderRead`, and with
/// `PixelFormatView`), both make it, answering the usage asked.
#[test]
fn written_swizzles_over_read_only_memory_are_refused() {
    use crate::os::OsSurface;
    use reims_vgpu_core::host::PhysicalMemoryRange;
    use reims_vgpu_core::num::ToU64 as _;
    use reims_vgpu_core::platform::os::surface::IoSurfaces as _;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::types::Swizzle;
    let vm = crate::os::OsVm::new();
    let page = vm.page_size();
    let guest = vm.allocate_anywhere(page).unwrap();
    let spec = SurfaceSpec {
        pixel_format: u32::from_be_bytes(*b"BGRA"),
        alloc_size: page,
        width: 64,
        height: 16,
        bytes_per_element: 4,
        bytes_per_row: 256,
        element_width: 1,
        element_height: 1,
        client_direction: Some(2),
        client_address_ranges: Some(vec![PhysicalMemoryRange {
            physical_address: guest.addr().to_u64(),
            physical_length: page,
        }]),
        ..SurfaceSpec::default()
    };
    let surface = crate::os::OsSurfaces::new().create(&spec).unwrap();
    assert!(crate::os::gpu_read_only(&surface));
    let desc = |usage, wse: bool| TextureDesc {
        pixel_format: 80,
        width: 64,
        height: 16,
        usage,
        resource_options: Some(0),
        swizzle: Some(Swizzle {
            red: 4,
            green: 3,
            blue: 2,
            alpha: 5,
        }),
        write_swizzle_enabled: wse.then_some(true),
        ..TextureDesc::default()
    };
    // Each creator's texture's usage, or `None` for its death, of `desc` on `d`: the
    // IOSurface creator's and a buffer's on the surface.
    fn made<P: Platform<Surface = OsSurface>>(
        d: &P::Device,
        surface: &OsSurface,
        desc: &P::TextureDescriptor,
    ) -> [u64; 2] {
        let buffer = d.new_buffer_on_surface(Some(surface)).unwrap().unwrap();
        let on_surface = d.new_texture_on_surface(desc, Some(surface), 0);
        let on_buffer = buffer.new_texture(OnFault::Return, desc, 0, 256);
        [on_surface, on_buffer].map(|t| t.unwrap().unwrap().usage())
    }
    let d = dev();
    let p = platform();
    let md = MetalPlatform::system_default_device();
    let metal = |usage, wse| reims_vgpu_metal::gpu::desc::texture_descriptor(&desc(usage, wse));
    for usage in [0x1, 0x11] {
        let ours = made::<VulkanPlatform>(&d, &surface, &p.texture_descriptor(&desc(usage, false)));
        assert_eq!(ours, [usage; 2], "usage {usage:#x}");
        if let Some(md) = &md {
            assert_eq!(
                made::<MetalPlatform>(md, &surface, &metal(usage, false)),
                ours
            );
        }
    }
    let buffer = d.new_buffer_on_surface(Some(&surface)).unwrap().unwrap();
    for usage in [0x3, 0x5] {
        for wse in [false, true] {
            let ours = p.texture_descriptor(&desc(usage, wse));
            let creators: [(&str, &dyn Fn()); 2] = [
                ("newTextureWithDescriptor:iosurface:plane:", &|| {
                    let _ = d.new_texture_on_surface(&ours, Some(&surface), 0);
                }),
                ("newTextureWithDescriptor:offset:bytesPerRow:", &|| {
                    let _ = buffer.new_texture(OnFault::Return, &ours, 0, 256);
                }),
            ];
            for (sel, make) in creators {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(make).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                let refused = match wse {
                    false => "Texture(SwizzleWithWrites)",
                    true => "not implemented in the runtime yet: WriteSwizzle",
                };
                assert!(line.contains(refused), "{usage:#x} {wse} {sel}: {line}");
            }
            if let (Some(md), true) = (&md, wse) {
                let theirs = made::<MetalPlatform>(md, &surface, &metal(usage, wse));
                assert_eq!(theirs, [usage; 2], "Metal: {usage:#x}");
            }
        }
    }
    drop((buffer, surface));
    assert!(vm.deallocate(guest, page).is_success());
}

/// A texture of `width` by `height` in `format`, of `ShaderRead` usage alone, over a Shared
/// buffer holding `bytes`, as a guest backing's texture is.
fn read_only_over_buffer(format: u64, width: u64, height: u64, bytes: &[u8]) -> VulkanTexture {
    let b = shared(width * height * 4);
    // SAFETY: the buffer's Shared bytes, which nothing else touches yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len()) };
    let desc = platform().texture_descriptor(&TextureDesc {
        pixel_format: format,
        width,
        height,
        usage: 0x1,
        ..TextureDesc::default()
    });
    b.new_texture(OnFault::Return, &desc, 0, width * 4)
        .unwrap()
        .unwrap()
}

/// A texture without PixelFormatView usage views as another format of its class, as Metal
/// makes it without API validation: an RGBA8Unorm texture over a buffer as BGRA8Unorm and
/// R32Uint, reading the same bytes; it and its views report the usage asked, Metal's
/// answers on this host
#[test]
fn views_need_no_pixel_format_view_usage() {
    let bytes = pattern(8, 4);
    let t = read_only_over_buffer(70, 8, 4, &bytes);
    let mut answers = Vec::new();
    for format in [80, 53] {
        let v = t.new_texture_view(format).unwrap().unwrap();
        assert_eq!(read_back(&v), bytes, "format {format}");
        answers.push((v.pixel_format(), v.usage()));
    }
    assert_eq!(t.usage(), 1);
    assert_eq!(answers, [(80, 1), (53, 1)]);
    let asked = d_texture(0x11);
    assert_eq!(asked.usage(), 0x11);
    assert_eq!(asked.new_texture_view(80).unwrap().unwrap().usage(), 0x11);
    assert_eq!(d_texture(0).usage(), 0x17, "Unknown, as Metal reports it");
    let handle = t.new_shared_texture_handle().unwrap();
    assert!(handle.is_none(), "a texture over a buffer is not shareable");
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
        pixel_format: 70,
        width: 8,
        height: 4,
        usage: 0x1,
        ..TextureDesc::default()
    });
    let m = metal.new_texture(OnFault::Return, &desc).unwrap().unwrap();
    let theirs: Vec<_> = [80, 53]
        .into_iter()
        .map(|format| {
            let v = m.new_texture_view(format).unwrap().unwrap();
            (v.pixel_format(), v.usage())
        })
        .collect();
    assert_eq!(theirs, answers);
}

/// How a command buffer reaches a surface ([`surface_references`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// A blit's copy of a texture on it into a buffer.
    Blit,
    /// A kernel's `read()` of a texture on it, bound at a slot.
    Kernel,
    /// A kernel's `read()` of a view of a texture on it, bound at a slot.
    View,
    /// A kernel's `read()` of a texture on it through an argument buffer holding its
    /// resource ID, made resident with `useResource:usage:`.
    Resident,
    /// A render pass that clears a texture on it and stores it.
    Attachment,
    /// A blit's copy of a buffer on it (`newBufferWithIOSurface:`) into another buffer.
    SurfaceBuffer,
    /// Not at all: a texture on it is made and released, and the command buffer's kernel
    /// reads a texture on another surface.
    Unused,
    /// A kernel's `read()` of a texture of the format (raw) over plane 0 of it, a two-plane
    /// surface of the OSType, bound at a slot.
    KernelYcbcr(u64, [u8; 4]),
    /// Not at all, as for `Unused`, the texture on it one of the format (raw) over plane 0
    /// of it, a two-plane surface of the OSType.
    UnusedYcbcr(u64, [u8; 4]),
    /// A draw's `sample()` of a texture on it, bound at a fragment slot, into a target on
    /// another surface.
    Sampled(Pipeline),
    /// A draw into a texture on it, a render pass's color attachment it clears and stores.
    Drawn(Pipeline),
}

/// The `VkPipeline` a draw of [`Reach::Sampled`] or [`Reach::Drawn`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pipeline {
    /// Made before the command buffer: the state drew the same in one before.
    Before,
    /// Made at the draw: the state's first draw, through a triangle fill mode of lines, a
    /// key its creation does not predict. The test draws on a device whose shader cache is
    /// its own and empty (`device_with_own_cache`), as a fresh machine's.
    AtTheDraw,
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceGetUseCount(buffer: *const std::ffi::c_void) -> i32;
}

/// `s`'s IOSurface's retain count (`CFGetRetainCount`, which core's unmap reads: Busy unless
/// 1) and its use count (`IOSurfaceGetUseCount`).
fn surface_counts(s: &OsSurface) -> (i64, i32) {
    let at =
        std::ptr::with_exposed_provenance::<std::ffi::c_void>(usize::try_from(s.id().0).unwrap());
    // SAFETY: a live IOSurfaceRef, which `s` holds.
    (s.retain_count(), unsafe { IOSurfaceGetUseCount(at) })
}

/// [`surface_counts`] of a 16 by 16 surface (`BGRA8Unorm`, or two planes for the YCbCr
/// reaches) as a command buffer of `d`'s reaches it as `reach` says: with the texture or
/// buffer on it made; once the command buffer committed waiting for an event, every other
/// handle to the texture or buffer released; after it completed, the command buffer still
/// held; and once the command buffer is released. Each step runs in an autorelease pool of
/// its own, popped before the counts are read. With them, how many pipelines `made` counts
/// were made from the command buffer's encoding until it completed.
fn surface_references<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    reach: Reach,
    made: &dyn Fn(&P::Device) -> u64,
) -> ([(i64, i32); 4], u64) {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};
    use reims_vgpu_core::rt::Pool;

    use crate::gpu::ycbcr::measure::{PROBE, sampler};
    use crate::gpu::ycbcr::measure::{Shape, desc as ycbcr_desc};
    let side = 16;
    let ycbcr = match reach {
        Reach::KernelYcbcr(format, ostype) | Reach::UnusedYcbcr(format, ostype) => {
            Some((format, ostype))
        }
        _ => None,
    };
    let s = match ycbcr {
        Some((_, ostype)) => Shape::of(&ostype).surface((side, side)),
        None => bgra_surface(side, side),
    };
    let other = bgra_surface(side, side);
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 5,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let on_s = match ycbcr {
        Some((format, _)) => p.texture_descriptor(&ycbcr_desc(format, (side, side))),
        None => desc.clone_ref(),
    };
    let q = d.new_command_queue().unwrap();
    let event = d.new_shared_event().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, side * side * 8, 0)
        .unwrap()
        .unwrap();
    let args = d
        .new_buffer_of_length(OnFault::Die, 64, 0)
        .unwrap()
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let state = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (read, held) = (state("read_texels"), state("read_held"));
    let elsewhere = d
        .new_texture_on_surface(&desc, Some(&other), 0)
        .unwrap()
        .unwrap();
    // A draw's state, sampler and probe (`sample_fragment`'s: each fragment's texel centre,
    // in rows of `side`).
    let render = |name| {
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let (vertex, fragment) = (function("cover_vertex"), function(name));
        d.new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: 80,
            resource_index: None,
        })
        .unwrap()
        .unwrap()
    };
    let drawing = match reach {
        Reach::Sampled(_) => Some(render("sample_fragment")),
        Reach::Drawn(_) => Some(render("gradient_fragment")),
        _ => None,
    };
    let nearest = sampler::<P>(d, (0, 0, false), 0);
    if let Reach::Sampled(_) = reach {
        let probe: [u32; 6] = [
            0.5f32.to_bits(),
            0.5f32.to_bits(),
            1f32.to_bits(),
            1f32.to_bits(),
            u32::try_from(side).unwrap(),
            u32::try_from(side).unwrap(),
        ];
        // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(probe.as_ptr(), args.contents().as_ptr::<u32>(), 6)
        };
    }
    let draw = |cb: &P::CommandBuffer, target: &P::Texture, sampled: Option<&P::Texture>, lines| {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(target)),
            load_action: Some(2),
            store_action: Some(1),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let buffers = [Some(&out), Some(&args)];
        let samplers = [Some(&nearest)];
        let mut cmds = vec![RenderCmd::SetRenderPipelineState {
            pipeline_state: drawing.as_ref(),
        }];
        if lines {
            cmds.push(RenderCmd::SetTriangleFillMode { fill_mode: 1 });
        }
        if let Some(t) = sampled {
            cmds.extend([
                RenderCmd::SetFragmentTexture {
                    texture: Some(t),
                    index: 0,
                },
                RenderCmd::SetFragmentSamplerStates {
                    samplers: &samplers,
                    range: NsRange::new(0, 1),
                },
                RenderCmd::SetFragmentBuffers {
                    buffers: &buffers,
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
            ]);
        }
        cmds.push(RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        });
        cmds.push(RenderCmd::EndEncoding);
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    };
    // The same draw before, into and of textures on no surface, completed and released.
    if let Reach::Sampled(Pipeline::Before) | Reach::Drawn(Pipeline::Before) = reach {
        let _pool = Pool::here();
        let plain = || d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let (target, sampled) = (plain(), plain());
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let sampled = matches!(reach, Reach::Sampled(_)).then_some(&sampled);
        draw(&cb, &target, sampled, false);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{reach:?}: {:?}", cb.error());
    }
    let made_before = made(d);
    let mut seen = [(0, 0); 4];
    let cb = {
        let _pool = Pool::here();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let (t, b) = match reach {
            Reach::SurfaceBuffer => (None, d.new_buffer_on_surface(Some(&s)).unwrap()),
            _ => (d.new_texture_on_surface(&on_s, Some(&s), 0).unwrap(), None),
        };
        assert!(
            t.is_some() || b.is_some(),
            "{reach:?}: nothing made on the surface"
        );
        seen[0] = surface_counts(&s);
        let view = match reach {
            Reach::View => t.as_ref().unwrap().new_texture_view(80).unwrap(),
            _ => None,
        };
        let kernel = |e: &P::ComputeEncoder, state, texture, resident: Option<&P::Texture>| {
            let buffers = [Some(&out), Some(&args)];
            let resources = [resident.map(ResourceRef::Texture)];
            let mut cmds = vec![
                ComputeCmd::SetComputePipelineState { state: Some(state) },
                ComputeCmd::SetBuffers {
                    buffers: &buffers,
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
            ];
            if let Some(texture) = texture {
                cmds.push(ComputeCmd::SetTexture {
                    texture: Some(texture),
                    index: 0,
                });
            }
            if resident.is_some() {
                cmds.push(ComputeCmd::UseResources {
                    resources: &resources,
                    usage: 1,
                });
            }
            cmds.push(ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(side, side, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            });
            cmds.push(ComputeCmd::EndEncoding);
            for cmd in cmds {
                e.encode(OnFault::Die, cmd).unwrap();
            }
        };
        match reach {
            Reach::Blit | Reach::SurfaceBuffer => {
                let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
                let copy = match &t {
                    Some(t) => BlitCmd::CopyTextureToBuffer {
                        source_texture: t,
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::new(0, 0, 0),
                        source_size: MtlSize::new(side, side, 1),
                        destination_buffer: &out,
                        destination_offset: 0,
                        destination_bytes_per_row: side * 4,
                        destination_bytes_per_image: side * side * 4,
                    },
                    None => BlitCmd::CopyBufferToBuffer {
                        source_buffer: b.as_ref(),
                        source_offset: 0,
                        destination_buffer: Some(&out),
                        destination_offset: 0,
                        size: side * side * 4,
                    },
                };
                e.encode(OnFault::Die, copy).unwrap();
                e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            }
            Reach::Attachment => {
                let pass = p.render_pass_descriptor();
                let attachment = ColorAttachmentSet {
                    texture: Some(t.as_ref()),
                    load_action: Some(2),
                    store_action: Some(1),
                    ..ColorAttachmentSet::default()
                };
                pass.set_color_attachment(OnFault::Die, 0, &attachment)
                    .unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            }
            Reach::Resident => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                let t = t.as_ref().unwrap();
                // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
                unsafe { args.contents().as_ptr::<u64>().write(t.gpu_resource_id()) };
                kernel(&e, &held, None, Some(t));
            }
            Reach::Kernel | Reach::KernelYcbcr(..) => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                kernel(&e, &read, t.as_ref(), None);
            }
            Reach::View => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                kernel(&e, &read, view.as_ref(), None);
            }
            Reach::Unused | Reach::UnusedYcbcr(..) => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                kernel(&e, &read, Some(&elsewhere), None);
            }
            Reach::Sampled(drawn) => {
                draw(&cb, &elsewhere, t.as_ref(), drawn == Pipeline::AtTheDraw);
            }
            Reach::Drawn(drawn) => {
                draw(&cb, t.as_ref().unwrap(), None, drawn == Pipeline::AtTheDraw);
            }
        }
        cb.commit(OnFault::Die).unwrap();
        cb
    };
    seen[1] = surface_counts(&s);
    {
        let _pool = Pool::here();
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{reach:?}: {:?}", cb.error());
    }
    seen[2] = surface_counts(&s);
    let made = made(d) - made_before;
    {
        let _pool = Pool::here();
        drop(cb);
    }
    seen[3] = surface_counts(&s);
    (seen, made)
}

/// A command buffer holds a surface's IOSurface as Metal's does, measured by its retain
/// count (which core's unmap reads: Busy unless 1) and its use count: from its commit until
/// it completed, through a texture on the surface that a blit copies, a kernel reads at a
/// slot or through an argument buffer made resident, a render pass stores, or a draw samples
/// at a fragment slot or draws into as its attachment (its `VkPipeline` made before the
/// command buffer, or at the draw), a buffer on it a blit copies, and a texture of
/// `YCBCR10_420_2P` (505) or `RGB10_420_2P` (543) over an 'x420' surface, or of
/// `RGB8_420_2P` (540) over a '420v' one, a kernel reads; not at all
/// through any of those textures, made and released, that it never names, whose retain
/// is back to 1 at once. A kernel's read of a view of the texture holds the IOSurface too;
/// Metal's view takes a reference of its own, 3 while the command buffer runs where this
/// platform's view shares its texture's, 2. Either way unmap answers Busy until the command
/// buffer completed. Neither platform's command buffer changes the use count
///
/// On this platform the kernel's read through the argument buffer made resident dies at the
/// dispatch naming `half-texel-conversion` ([`crate::answer::DIES`]): the runtime announces the
/// kind for a `half` read of an argument buffer's texture, whose format the function does not
/// know and whose texel it converts by a rounding Metal's may not be. That case runs last here.
///
/// Metal's side runs every case first, before this platform makes anything in the process:
/// once MoltenVK has worked on Metal's device, Metal lets go of what a completed command
/// buffer held late (`texture_identifiers_are_reused_as_metals`)
#[test]
fn a_command_buffer_holds_a_surface_as_metals_does() {
    let ycbcr = [(505, *b"x420"), (543, *b"x420"), (540, *b"420v")];
    let named = [
        Reach::Blit,
        Reach::Kernel,
        Reach::Resident,
        Reach::Attachment,
        Reach::SurfaceBuffer,
        Reach::Sampled(Pipeline::Before),
        Reach::Sampled(Pipeline::AtTheDraw),
        Reach::Drawn(Pipeline::Before),
        Reach::Drawn(Pipeline::AtTheDraw),
    ];
    let named = named
        .into_iter()
        .chain(ycbcr.map(|(f, o)| Reach::KernelYcbcr(f, o)));
    let unnamed = [Reach::Unused]
        .into_iter()
        .chain(ycbcr.map(|(f, o)| Reach::UnusedYcbcr(f, o)));
    let cases: Vec<_> = named
        .map(|reach| (reach, [(2, 0), (2, 0), (1, 0), (1, 0)]))
        .chain(unnamed.map(|reach| (reach, [(2, 0), (1, 0), (1, 0), (1, 0)])))
        .chain([(Reach::View, [(2, 0), (2, 0), (1, 0), (1, 0)])])
        .collect();
    let theirs: Option<Vec<_>> = MetalPlatform::system_default_device().map(|md| {
        let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
        (cases.iter())
            .map(|&(reach, _)| {
                let (mut theirs, _) = surface_references::<MetalPlatform>(&mp, &md, reach, &|_| 0);
                if reach == Reach::View {
                    assert_eq!(
                        theirs[1],
                        (3, 0),
                        "Metal's view holds a reference of its own"
                    );
                    theirs[1].0 -= 1;
                }
                theirs
            })
            .collect()
    });
    let (p, d) = (platform(), dev());
    let made = |d: &VulkanDevice| d.raw.work_counts().variant_pipelines_made;
    // The draws' device: a shader cache of its own, empty, as a fresh machine's.
    let (own, _cache) = crate::gpu::device::tests::metal_tests::device_with_own_cache();
    for (i, &(reach, want)) in cases.iter().enumerate() {
        if reach == Reach::Resident {
            continue;
        }
        let on = match reach {
            Reach::Sampled(_) | Reach::Drawn(_) => &own,
            _ => &d,
        };
        let (ours, made) =
            surface_references::<crate::platform::VulkanPlatform>(&p, on, reach, &made);
        assert_eq!(ours, want, "{reach:?}");
        match reach {
            Reach::Sampled(Pipeline::AtTheDraw) | Reach::Drawn(Pipeline::AtTheDraw) => {
                assert_eq!(made, 1, "{reach:?}: its pipeline made at the draw");
            }
            Reach::Sampled(_) | Reach::Drawn(_) => {
                assert_eq!(made, 0, "{reach:?}: its pipeline made before");
            }
            _ => {}
        }
        if let Some(theirs) = &theirs {
            assert_eq!(ours, theirs[i], "{reach:?}: ours and Metal's");
        }
    }
    let kind = metal2vulkan_runtime::Degradation::HalfTexelConversion;
    let site = crate::answer::dies_of(kind, || {
        surface_references::<crate::platform::VulkanPlatform>(&p, &d, Reach::Resident, &made)
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
    // The device before the directory of its shader cache, which it uses while it lives: a
    // pattern's bindings drop last first.
    drop(own);
}

/// What holds a surface beside a command buffer's own uses of it ([`surface_holds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beside {
    /// Another queue's command buffer, committed after a kernel read a texture on it and
    /// before the texture's release, which reads a texture on another surface: counts once
    /// the texture is released, the other command buffer waiting for an event; once it
    /// completed; once it is released.
    Pending,
    /// As `Pending`, the other command buffer's kernel reading its texture through an
    /// argument buffer holding its resource ID, made resident with `useResource:usage:`.
    PendingArgument,
    /// As `Pending`, the other command buffer's kernel reaching a buffer of the device's
    /// own through an address it loads (`reach.metal`), naming no texture.
    PendingReach,
    /// A completed handler of the command buffer whose kernel read a texture on it, the
    /// texture released before the commit: counts inside the handler as it begins and
    /// 100 ms later; once the wait for the completion returned; 200 ms later; once the
    /// command buffer is released.
    Handler,
    /// Two textures on it, the first read by a kernel, the second never used: counts once
    /// the command buffer completed and is released; once the first is released; once
    /// the second is.
    Two,
}

/// [`surface_counts`] of a 16 by 16 `BGRA8Unorm` surface at the steps `beside` names, each
/// step in an autorelease pool of its own.
fn surface_holds<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    beside: Beside,
) -> Vec<(i64, i32)> {
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use reims_vgpu_core::rt::Pool;

    use crate::gpu::ycbcr::measure::PROBE;
    let side = 16;
    let s = Arc::new(bgra_surface(side, side));
    let other = bgra_surface(side, side);
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 5,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let (q, q2) = (
        d.new_command_queue().unwrap(),
        d.new_command_queue().unwrap(),
    );
    let event = d.new_shared_event().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, side * side * 8, 0)
        .unwrap()
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let state = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (read, held) = (state("read_texels"), state("read_held"));
    let args = d
        .new_buffer_of_length(OnFault::Die, 64, 0)
        .unwrap()
        .unwrap();
    let elsewhere = d
        .new_texture_on_surface(&desc, Some(&other), 0)
        .unwrap()
        .unwrap();
    let reading_how = |q: &P::CommandQueue, texture: &P::Texture, wait: bool, argument: bool| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        if wait {
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        }
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&out), Some(&args)];
        let resources = [Some(ResourceRef::Texture(texture))];
        let mut cmds = vec![
            ComputeCmd::SetComputePipelineState {
                state: Some(match argument {
                    true => &held,
                    false => &read,
                }),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
        ];
        if argument {
            // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
            unsafe {
                args.contents()
                    .as_ptr::<u64>()
                    .write(texture.gpu_resource_id())
            };
            cmds.push(ComputeCmd::UseResources {
                resources: &resources,
                usage: 1,
            });
        } else {
            cmds.push(ComputeCmd::SetTexture {
                texture: Some(texture),
                index: 0,
            });
        }
        for cmd in cmds.into_iter().chain([
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(side, side, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            },
            ComputeCmd::EndEncoding,
        ]) {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb
    };
    let reading = |q, texture, wait| reading_how(q, texture, wait, false);
    // The command buffer, and the buffer its kernel reaches through the address it loads,
    // which no command names, so the caller holds it until the command buffer completed.
    let reaching = |q: &P::CommandQueue| {
        const REACH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reach.metallib"));
        let library = d.new_library(REACH).unwrap().unwrap().unwrap();
        let f = library
            .new_function(OnFault::Die, "reach")
            .unwrap()
            .unwrap();
        let state = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let page = || {
            d.new_buffer_of_length(OnFault::Die, 4096, 0)
                .unwrap()
                .unwrap()
        };
        let (addresses, words, reached) = (page(), page(), page());
        // SAFETY: a Shared buffer of a page, which no command uses yet.
        unsafe {
            addresses
                .contents()
                .as_ptr::<u64>()
                .write(reached.gpu_address())
        };
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&addresses), Some(&words)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        (cb, reached)
    };
    let done = |cb: &P::CommandBuffer| {
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{beside:?}: {:?}", cb.error());
    };
    let on_s = || {
        d.new_texture_on_surface(&desc, Some(&s), 0)
            .unwrap()
            .unwrap()
    };
    let mut seen = Vec::new();
    match beside {
        Beside::Pending | Beside::PendingArgument | Beside::PendingReach => {
            let t = {
                let _pool = Pool::here();
                on_s()
            };
            {
                let _pool = Pool::here();
                let cb = reading(&q, &t, false);
                cb.commit(OnFault::Die).unwrap();
                done(&cb);
            }
            let (cb, reached) = {
                let _pool = Pool::here();
                let argument = beside == Beside::PendingArgument;
                let (cb, reached) = match beside {
                    Beside::PendingReach => {
                        let (cb, reached) = reaching(&q2);
                        (cb, Some(reached))
                    }
                    _ => (reading_how(&q2, &elsewhere, true, argument), None),
                };
                cb.commit(OnFault::Die).unwrap();
                (cb, reached)
            };
            {
                let _pool = Pool::here();
                drop(t);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                event.set_signaled_value(1).unwrap();
                done(&cb);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(cb);
                drop(reached);
            }
            seen.push(surface_counts(&s));
        }
        Beside::Handler => {
            let inside = Arc::new(Mutex::new(Vec::new()));
            let cb = {
                let _pool = Pool::here();
                let t = on_s();
                let cb = reading(&q, &t, false);
                drop(t);
                let (s, inside) = (Arc::clone(&s), Arc::clone(&inside));
                let handler = move |_| {
                    inside.lock().push(surface_counts(&s));
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    inside.lock().push(surface_counts(&s));
                };
                cb.add_completed_handler(OnFault::Die, Box::new(handler))
                    .unwrap();
                cb.commit(OnFault::Die).unwrap();
                cb
            };
            {
                let _pool = Pool::here();
                done(&cb);
            }
            let after = surface_counts(&s);
            std::thread::sleep(std::time::Duration::from_millis(200));
            seen.extend(inside.lock().drain(..));
            seen.push(after);
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(cb);
            }
            seen.push(surface_counts(&s));
        }
        Beside::Two => {
            let (t, unused) = {
                let _pool = Pool::here();
                (on_s(), on_s())
            };
            {
                let _pool = Pool::here();
                let cb = reading(&q, &t, false);
                cb.commit(OnFault::Die).unwrap();
                done(&cb);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(t);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(unused);
            }
            seen.push(surface_counts(&s));
        }
    }
    seen
}

/// [`surface_holds`] of each case on this platform and on Metal's, which agree, and are its
/// `want`. Metal's side runs every case first, before this platform makes anything in the
/// process: once MoltenVK has worked on Metal's device, Metal lets go of what a completed
/// command buffer held late (`texture_identifiers_are_reused_as_metals`).
fn holds_as_metals(cases: &[(Beside, &[(i64, i32)])]) {
    let theirs: Option<Vec<_>> = MetalPlatform::system_default_device().map(|md| {
        let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
        (cases.iter())
            .map(|&(beside, _)| surface_holds::<MetalPlatform>(&mp, &md, beside))
            .collect()
    });
    let (p, d) = (platform(), dev());
    for (i, &(beside, want)) in cases.iter().enumerate() {
        let ours = surface_holds::<crate::platform::VulkanPlatform>(&p, &d, beside);
        assert_eq!(ours, want, "{beside:?}");
        if let Some(theirs) = &theirs {
            assert_eq!(ours, theirs[i], "{beside:?}: ours and Metal's");
        }
    }
}

/// Nothing beside a command buffer's own uses holds a surface's IOSurface, as on Metal: not
/// a command buffer of another queue that never names it, waiting while the texture on it
/// a completed one read is released, whose kernel reads a texture at a slot; not a second
/// texture on it, whose release lets go of its own reference only. A command buffer's
/// completed handler runs while it still holds the IOSurface, on Metal as here, and lets it go
/// once the handler returned. Where that other command buffer's kernel reads its texture
/// through an argument buffer, Metal holds nothing either, and this platform dies at the
/// dispatch naming `half-texel-conversion` ([`crate::answer::DIES`]): the runtime announces the
/// kind for a `half` read of an argument buffer's texture, whose format the function does not
/// know and whose texel it converts by a rounding Metal's may not be. Metal's side of it runs
/// first
#[test]
fn nothing_beside_a_command_buffer_holds_a_surface() {
    let none = [(1, 0); 3];
    let handled = [(2, 0), (2, 0), (1, 0), (1, 0), (1, 0)];
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
        let theirs = surface_holds::<MetalPlatform>(&mp, &md, Beside::PendingArgument);
        assert_eq!(theirs, none, "Metal's, through an argument buffer");
    }
    holds_as_metals(&[
        (Beside::Pending, &none[..]),
        (Beside::Handler, &handled[..]),
        (Beside::Two, &[(3, 0), (2, 0), (1, 0)][..]),
    ]);
    let kind = metal2vulkan_runtime::Degradation::HalfTexelConversion;
    let site = crate::answer::dies_of(kind, || {
        surface_holds::<crate::platform::VulkanPlatform>(
            &platform(),
            &dev(),
            Beside::PendingArgument,
        )
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// A command buffer of another queue that never names a surface, waiting while the texture
/// on it a completed one read is released, holds nothing of it where its kernel reaches a
/// buffer through an address it loads, as on Metal
#[test]
fn a_command_buffer_reaching_an_address_holds_no_surface_it_never_names() {
    holds_as_metals(&[(Beside::PendingReach, &[(1, 0); 3][..])]);
}

/// The address of a one-page buffer made and released on `d`: where the next range lies.
pub(crate) fn next_range<P: Platform>(d: &P::Device) -> u64 {
    use reims_vgpu_core::platform::resource::Buffer as _;
    let buffer = d.new_buffer_of_length(OnFault::Die, 0x4000, 0);
    buffer.unwrap().unwrap().gpu_address()
}

/// How far the next range moves ([`next_range`]) as up to `count` objects `make` makes are
/// made and kept on `d`, move by move; and whether it is back where it began once they are
/// all released.
fn range_moves<P: Platform, T>(
    d: &P::Device,
    count: usize,
    make: &dyn Fn() -> T,
) -> (Vec<u64>, bool) {
    let first = next_range::<P>(d);
    let (mut was, mut kept, mut moved) = (first, Vec::new(), Vec::new());
    for _ in 0..count {
        kept.push(make());
        let now = next_range::<P>(d);
        if now != was {
            moved.push(now.wrapping_sub(was));
            was = now;
        }
    }
    drop(kept);
    (moved, next_range::<P>(d) == first)
}

/// Private 32x64 RGBA8 textures read by shaders, of a heap size of 0x2000 bytes or a little
/// more, and private buffers of 0x100 bytes take slots in Metal's private pool, under 16 and
/// 512 to a pool, so one of the first 16 textures, and one of the first 513 buffers, makes
/// another pool, which takes a range of 0x28000 and goes once they are released
/// (`metal_space::PoolClass`): on Metal, where it runs, the next range moves on by that once
/// and comes back; here, the model's pools do. The runtime answers Metal's heap size, and the
/// textures' slots lie that apart, rounded up to 0x100, within the first pool, the last at
/// the next pool's first byte
#[test]
fn small_private_textures_and_buffers_fill_metals_private_pool() {
    use crate::gpu::metal_space::{PoolClass, process};
    let mut metal_heap = None;
    let rgba8 = TextureDesc {
        pixel_format: 70,
        width: 32,
        height: 64,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    };
    if let Some(md) = MetalPlatform::system_default_device() {
        // A queue makes the pools.
        drop(md.new_command_queue().unwrap());
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&rgba8);
        let texture = || md.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let buffer = || {
            md.new_buffer_of_length(OnFault::Die, 0x100, 0x20)
                .unwrap()
                .unwrap()
        };
        let textures = range_moves::<MetalPlatform, _>(&md, 16, &texture);
        let buffers = range_moves::<MetalPlatform, _>(&md, 513, &buffer);
        let once = (vec![0x2_8000], true);
        assert_eq!([textures, buffers], [once.clone(), once], "Metal");
        metal_heap = Some(md.heap_texture_size_and_align(&desc).unwrap().size);
    }
    let (p, d) = (platform(), dev());
    // A queue makes the pools, as on Metal: the textures' first one is not the first use.
    drop(d.new_command_queue().unwrap());
    let pools = || process().pools_of(PoolClass::Private);
    let before = pools();
    let desc = p.texture_descriptor(&rgba8);
    let textures: Vec<_> = (0..16)
        .map(|_| d.new_texture(OnFault::Die, &desc).unwrap().unwrap())
        .collect();
    assert_eq!(pools(), before + 1, "16 textures");
    let heap = d.heap_texture_size_and_align(&desc).unwrap().size;
    if let Some(metal) = metal_heap {
        assert_eq!(heap, metal, "the heap size");
    }
    let step = heap.next_multiple_of(0x100);
    let slots: Vec<u64> = textures
        .iter()
        .map(|t| t.space.address().unwrap())
        .collect();
    let (first, last) = slots.split_at(15);
    assert!(
        first.windows(2).all(|pair| pair[1] == pair[0] + step)
            && last[0] != first[14] + step
            && last[0] % crate::gpu::metal_space::SLOT == 0,
        "{step:#x}: {slots:x?}"
    );
    drop(textures);
    assert_eq!(pools(), before, "the textures released");
    let buffers: Vec<_> = (0..513)
        .map(|_| {
            d.new_buffer_of_length(OnFault::Die, 0x100, 0x20)
                .unwrap()
                .unwrap()
        })
        .collect();
    assert_eq!(pools(), before + 1, "513 buffers");
    drop(buffers);
    assert_eq!(pools(), before, "the buffers released");
}

/// Buffers under a page of each storage mode lie at their slots in Metal's pools
/// (`metal_space::PoolClass`): after a queue, ones of 1, 0x100, 0x101, 0x1001 and 0x3fff
/// bytes take 0x100, 0x100, 0x200, 0x1100 and 0x4000 bytes, each right after the one before,
/// the first shared one 0x4500 bytes into its pool, the first private one 0x100 and the first
/// managed one at a new pool's first byte, and one of 0x80 bytes made after the second's
/// release takes its slot. On Metal, where it runs, each lies that far from the first, and
/// the first that far into a 32-KiB granule; here, each answers its pool's address in the
/// space's model plus that offset, the runtime's slot in its reservation of the pool.
/// Another device's shared buffers lie after the first's in the same pool, in a reservation
/// of its own. The managed pool goes with its last buffer, and comes back at the next
#[test]
fn small_buffers_lie_at_their_slots_in_metals_pools() {
    use reims_vgpu_core::platform::resource::Buffer as _;

    use crate::gpu::metal_space::{PoolClass, SLOT, process};
    const LENGTHS: [u64; 5] = [1, 0x100, 0x101, 0x1001, 0x3fff];
    const FROM_FIRST: [u64; 6] = [0, 0x100, 0x200, 0x400, 0x1500, 0x100];
    let classes = [
        (PoolClass::Shared, 0, 0x4500),
        (PoolClass::Private, 0x20, 0x100),
        (PoolClass::Managed, 0x10, 0),
    ];
    fn made<P: Platform>(d: &P::Device, options: u64) -> (Vec<P::Buffer>, Vec<u64>) {
        let make = |length| {
            let buffer = d.new_buffer_of_length(OnFault::Die, length, options);
            buffer.unwrap().unwrap()
        };
        let mut buffers: Vec<_> = LENGTHS.into_iter().map(make).collect();
        let mut at: Vec<u64> = buffers.iter().map(|b| b.gpu_address()).collect();
        drop(buffers.remove(1));
        buffers.push(make(0x80));
        at.extend(buffers.last().map(|b| b.gpu_address()));
        (buffers, at)
    }
    let mut metal = Vec::new();
    if let Some(md) = MetalPlatform::system_default_device() {
        drop(md.new_command_queue().unwrap());
        for (_, options, _) in classes {
            let (_kept, at) = made::<MetalPlatform>(&md, options);
            let first = at[0];
            let from_first: Vec<u64> = at.iter().map(|a| a.wrapping_sub(first)).collect();
            metal.push((first % SLOT, from_first));
        }
    }
    let d = dev();
    drop(d.new_command_queue().unwrap());
    let managed = || process().pool_addresses(PoolClass::Managed);
    let before = managed();
    let mut kept = Vec::new();
    for (class, options, offset) in classes {
        let (buffers, at) = made::<VulkanPlatform>(&d, options);
        let pool = *process().pool_addresses(class).last().unwrap();
        let into: Vec<u64> = at.iter().map(|a| a.wrapping_sub(pool)).collect();
        let expected: Vec<u64> = FROM_FIRST.iter().map(|o| offset + o).collect();
        assert_eq!(
            into, expected,
            "{class:?} at {at:x?}, its pool at {pool:#x}"
        );
        kept.push(buffers);
    }
    assert_eq!(managed().len(), before.len() + 1, "the managed pool");
    // Another device of the process places its buffers in the same pools, after the first
    // device's, as Metal's one device does: in the runtime's reservation of its own.
    let (own, _cache) = crate::gpu::device::tests::metal_tests::device_with_own_cache();
    let (others, at) = made::<VulkanPlatform>(&own, 0);
    let pool = process().pool_addresses(PoolClass::Shared)[0];
    let into: Vec<u64> = at.iter().map(|a| a.wrapping_sub(pool)).collect();
    let expected: Vec<u64> = FROM_FIRST.iter().map(|o| 0x9a00 + o).collect();
    assert_eq!(into, expected, "another device, at {at:x?}");
    drop((kept, others));
    assert_eq!(managed(), before, "the managed pool, its buffers released");
    let (_again, at) = made::<VulkanPlatform>(&d, 0x10);
    let pool = *managed().last().unwrap();
    assert_eq!(at[0], pool, "the managed pool again");
    for (row, (_, _, offset)) in metal.into_iter().zip(classes) {
        assert_eq!(row, (offset, FROM_FIRST.to_vec()), "Metal");
    }
}

/// Sampler states of argument buffer support, each of its own LOD clamp, take descriptors in
/// Metal's sampler mempool, whose array takes a range at the 255th after a queue, the
/// device's own sampler counted (`metal_space::Samplers`): on Metal, where it runs, and
/// here, the next range moves on by 0x8000 there, once in 300
#[test]
fn sampler_states_grow_metals_sampler_mempool() {
    let made = std::cell::Cell::new(0u16);
    let next = || {
        made.set(made.get() + 1);
        SamplerDesc {
            support_argument_buffers: Some(true),
            lod_max_clamp: Some(1000.0 + f32::from(made.get())),
            ..SamplerDesc::default()
        }
    };
    if let Some(md) = MetalPlatform::system_default_device() {
        drop(md.new_command_queue().unwrap());
        let make = || md.new_sampler_state(&next()).unwrap().unwrap();
        let (moves, _) = range_moves::<MetalPlatform, _>(&md, 300, &make);
        assert_eq!(moves, [0x8000], "Metal");
    }
    let d = dev();
    drop(d.new_command_queue().unwrap());
    let make = || d.new_sampler_state(&next()).unwrap().unwrap();
    let (moves, _) = range_moves::<VulkanPlatform, _>(&d, 300, &make);
    assert_eq!(moves, [0x8000]);
}

/// A view in a value Metal describes no format for is nil, as Metal answers, from each of the
/// six view selectors, every value from 0 to 2047 tried, `MTLPixelFormatInvalid`'s 0 and the
/// formats of a later SDK (45 to 48, 95 to 100, 120 to 122) among them; before its swizzle,
/// type, levels and slices are checked, as Metal's: a swizzle channel of 9, type 99 and
/// levels whose end passes `u64::MAX` (measured one case a process, from `BGRA8Unorm`,
/// `RGBA16Float`, `R16Float` and `Depth32Float_Stencil8` sources, of `ShaderRead` usage with
/// and without `PixelFormatView`; compared here with the three selectors that take no resource
/// index). But for 255 and 262, which Metal's setter takes: a view in either, of the
/// texture's type, levels and swizzle, dies at its selector, where Metal asserts
/// ("_mtlValidateArgumentsForTextureViewOnDevice:1866: failed assertion `Texture Creation
/// texture view pixelFormat (MTLPixelFormatInvalid) not castable."; a swizzle channel of 9
/// asserts first)
#[test]
fn views_metal_answers_nil_are_nil() {
    use reims_vgpu_core::platform::resource::Texture as _;
    let t = dev()
        .new_texture(OnFault::Return, &bgra(8, 8, 0))
        .unwrap()
        .unwrap();
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let theirs = md.as_ref().map(|md| {
        let d = mp.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 8,
            height: 8,
            usage: 0x13,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        md.new_texture(OnFault::Return, &d).unwrap().unwrap()
    });
    let r = |location, length| NsRange { location, length };
    let (one, past) = (r(0, 1), r(u64::MAX, 2));
    let nine = Swizzle {
        red: 9,
        green: 3,
        blue: 4,
        alpha: 5,
    };
    let plain = Swizzle {
        red: 2,
        green: 3,
        blue: 4,
        alpha: 5,
    };
    type View<'a> = (&'static str, Box<dyn Fn() -> Option<VulkanTexture> + 'a>);
    // Each selector's view of `t` in `f`, the subviews of a swizzle, type or levels Metal
    // checks after the format where `odd`, else of the texture's.
    let views = |f: u64, odd: bool| -> [View<'_>; 6] {
        let t = &t;
        let (kind, levels, nine) = match odd {
            true => (99, past, nine),
            false => (2, one, plain),
        };
        [
            (
                "newTextureViewWithPixelFormat:",
                Box::new(move || t.new_texture_view(f).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:resourceIndex:",
                Box::new(move || t.new_texture_view_at_index(f, 0).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:",
                Box::new(move || t.new_texture_subview(f, kind, levels, one).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
                Box::new(move || (t.new_swizzled_texture_subview(f, 2, one, one, nine)).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
                Box::new(move || {
                    (t.new_swizzled_texture_subview_at_index(f, kind, one, one, nine, 0)).unwrap()
                }),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
                Box::new(move || (t.new_texture_subview_at_index(f, 2, levels, one, 0)).unwrap()),
            ),
        ]
    };
    let mut nil = 0;
    for f in (0..2048)
        .filter(|&f| !crate::gpu::format_info::described("newTextureViewWithPixelFormat:", f))
    {
        let taken = matches!(f, 255 | 262);
        for (sel, view) in views(f, !taken) {
            if taken {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(&view).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!(
                    (site.kind, site.file),
                    (DeathKind::Abort, sel),
                    "{f}: {line}"
                );
                assert!(line.contains(&format!("pixel format {f}")), "{line}");
                assert!(line.contains("Metal aborts at"), "{line}");
            } else {
                assert!(view().is_none(), "{sel} {f}");
                nil += 1;
            }
        }
        let Some(theirs) = (theirs.as_ref()).filter(|_| !matches!(f, 255 | 262)) else {
            continue;
        };
        assert!(theirs.new_texture_view(f).unwrap().is_none(), "Metal {f}");
        let sub = theirs.new_texture_subview(f, 99, past, one).unwrap();
        assert!(sub.is_none(), "Metal {f}");
        let swizzled = theirs
            .new_swizzled_texture_subview(f, 2, one, one, nine)
            .unwrap();
        assert!(swizzled.is_none(), "Metal {f}");
    }
    assert_eq!(nil, 6 * (2048 - 264 - 2));
}

/// The surface's bytes, after the work that wrote them completed.
fn surface_bytes(s: &OsSurface) -> Vec<u8> {
    let n = usize::try_from(s.alloc_size()).unwrap();
    // SAFETY: the surface's memory, which no command uses now.
    unsafe { std::slice::from_raw_parts(s.base_address().as_ptr::<u8>(), n) }.to_vec()
}

/// What a render pass of [`rotated_uses`] draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drawn {
    /// Nothing: it sets its pipeline state alone.
    Nothing,
    /// A gradient over the top left half of its frame, which reaches every 32 by 32 tile of
    /// a frame of 32 by 32.
    Corner,
    /// That gradient under a scissor rectangle of the frame's top left 32 by 32 tile, which
    /// reaches that tile alone of a frame of 64 by 64.
    FirstTile,
}

impl Drawn {
    /// The side of the square frame the pass draws: 64 for [`Drawn::FirstTile`], else 32.
    fn side(self) -> u64 {
        match self {
            Drawn::FirstTile => 64,
            Drawn::Nothing | Drawn::Corner => 32,
        }
    }
}

/// What a kernel, a blit, the CPU and a render pass see and leave of a texture of `rotation`
/// over a square `BGRA8Unorm` surface of `drawn`'s side whose texels name their places, on
/// `p`'s device `d`, each named: a kernel's `read()` of every texel, a blit's copy of the
/// texture into a buffer, the surface's bytes after the CPU replaced an 8 by 4 region at
/// (4, 2) of the texture, the surface's bytes after a pass of load action `load` that draws
/// `drawn`, and a kernel's `read()` after the pass.
fn rotated_uses<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    rotation: u64,
    (load, drawn): (u64, Drawn),
) -> Vec<(&'static str, Vec<u8>)> {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};

    use crate::gpu::ycbcr::measure::{PROBE, probed};
    let side = drawn.side();
    let s = bgra_surface(side, side);
    // Each texel names its place: (x, y, 7, 200).
    let fill: Vec<u8> = (0..side * side)
        .flat_map(|i| [(i % side) as u8, (i / side) as u8, 7, 200])
        .collect();
    // SAFETY: the surface's memory, of `side` rows of `side` texels, which no command uses.
    unsafe {
        std::ptr::copy_nonoverlapping(fill.as_ptr(), s.base_address().as_ptr::<u8>(), fill.len());
    }
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 5,
        resource_options: Some(0),
        framebuffer_only: Some(true),
        rotation: Some(rotation),
        ..TextureDesc::default()
    });
    let t = d
        .new_texture_on_surface(&desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    let mut seen = vec![(
        "read",
        probed::<P>(d, "read_texels", &t, (side, side), None, None),
    )];
    let q = d.new_command_queue().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, side * side * 4, 0)
        .unwrap()
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let copy = BlitCmd::CopyTextureToBuffer {
        source_texture: &t,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(side, side, 1),
        destination_buffer: &out,
        destination_offset: 0,
        destination_bytes_per_row: side * 4,
        destination_bytes_per_image: side * side * 4,
    };
    e.encode(OnFault::Die, copy).unwrap();
    e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let n = usize::try_from(side * side * 4).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    let copied = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
    seen.push(("blit", copied.to_vec()));
    let region = MtlRegion::new(MtlOrigin::new(4, 2, 0), MtlSize::new(8, 4, 1));
    let bytes: Vec<u8> = (0..8 * 4 * 4).map(|i: u8| i.wrapping_mul(3) | 1).collect();
    t.replace_region(region, 0, 0, HostRange::of_slice(&bytes), 8 * 4, 0)
        .unwrap();
    seen.push(("cpu", surface_bytes(&s)));
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function("corner_vertex"), function("gradient_fragment"));
    let state = d
        .new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: 80,
            resource_index: None,
        })
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&t)),
        load_action: Some(load),
        store_action: Some(1),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    let set = RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(&state),
    };
    let scissor = RenderCmd::SetScissorRect {
        rect: reims_vgpu_core::platform::types::MtlScissorRect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        },
    };
    let draw = RenderCmd::DrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
    };
    let cmds = match drawn {
        Drawn::Nothing => vec![set],
        Drawn::Corner => vec![set, draw],
        Drawn::FirstTile => vec![set, scissor, draw],
    };
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    seen.push(("pass", surface_bytes(&s)));
    seen.push((
        "read after the pass",
        probed::<P>(d, "read_texels", &t, (side, side), None, None),
    ));
    seen
}

/// A texture of each rotation over an IOSurface reads, copies, takes the CPU's writes and
/// stores a pass's texels as Metal's does, the surface's bytes compared byte for byte: a
/// kernel's `read()` sees the stored frame unturned, a blit and the CPU address the plane at
/// its rows, and a pass that clears or loads and draws over every 32 by 32 tile of its frame
/// stores its texels turned (rotation 2 by 180 degrees, 1 and 4 a quarter turn, 3 the other),
/// one that loads loading them turned back, but at rotation 4, which loads the plane
/// transposed, so the texels it loaded land mirrored left to right (the plane's texel (0, 0)
/// holding what (31, 0) held). Each rotation's pass that draws leaves the surface otherwise
/// than rotation 0's. A pass that clears or loads and draws nothing stores the plane as
/// Metal's does: in place where it loads. The pass that loads a texture of rotation 4 and
/// draws dies at `endEncoding` naming `rotated-load-tiles` ([`crate::answer::DIES`]), which the
/// runtime announces at every such pass, though this one's draw reaches every tile; Metal's
/// side of it runs
#[test]
fn rotated_textures_read_copy_and_store_as_metals() {
    let (p, d) = (platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let passes = [CLEAR, LOAD].map(|l| [(l, Drawn::Corner), (l, Drawn::Nothing)]);
    for pass in passes.into_iter().flatten() {
        let unturned = rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, 0, pass);
        for rotation in 1..=4 {
            let case = format!("rotation {rotation} (load action, drawn) {pass:?}");
            if (rotation, pass) == (4, (LOAD, Drawn::Corner)) {
                if let Some(md) = &md {
                    let theirs = rotated_uses::<MetalPlatform>(&mp, md, rotation, pass);
                    assert_eq!(theirs[3].1[..4], [31, 0, 7, 200], "{case}: Metal's mirror");
                }
                let kind = rt::Degradation::RotatedLoadTiles;
                let site = crate::answer::dies_of(kind, || {
                    rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, rotation, pass)
                });
                assert_eq!(site.file, "endEncoding", "{case}");
                continue;
            }
            let ours = rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, rotation, pass);
            if pass.1 == Drawn::Corner {
                assert_ne!(ours[3], unturned[3], "{case}: the pass turns nothing");
            }
            let Some(md) = &md else { continue };
            let theirs = rotated_uses::<MetalPlatform>(&mp, md, rotation, pass);
            for ((what, o), (_, m)) in ours.iter().zip(&theirs) {
                let differ = o.chunks(4).zip(m.chunks(4)).filter(|(a, b)| a != b).count();
                assert_eq!(differ, 0, "{case}: {what}");
            }
        }
    }
}

/// A texture of `rotation` over a square `BGRA8Unorm` surface of `side` whose texels name
/// their places (as [`rotated_uses`] fills it), made on `p`'s device `d` with `ShaderRead`,
/// `RenderTarget` and `PixelFormatView` usage, and the surface.
fn rotated_named<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    side: u64,
    rotation: u64,
) -> (OsSurface, P::Texture) {
    let s = bgra_surface(side, side);
    let fill: Vec<u8> = (0..side * side)
        .flat_map(|i| [(i % side) as u8, (i / side) as u8, 7, 200])
        .collect();
    // SAFETY: the surface's memory, of `side` rows of `side` texels, which no command uses.
    unsafe {
        std::ptr::copy_nonoverlapping(fill.as_ptr(), s.base_address().as_ptr::<u8>(), fill.len());
    }
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 0x15,
        resource_options: Some(0),
        framebuffer_only: Some(true),
        rotation: Some(rotation),
        ..TextureDesc::default()
    });
    let t = d
        .new_texture_on_surface(&desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    (s, t)
}

/// The surface's bytes after a pass of load action `load` that draws `drawn` (as
/// [`rotated_uses`]'s pass does) through a view in `view` of a texture of `rotation`
/// ([`rotated_named`]), its pipeline state of the view's format, on `p`'s device `d`.
fn rotated_view_pass<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    rotation: u64,
    view: u64,
    (load, drawn): (u64, Drawn),
) -> Vec<u8> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Library as _, Texture as _};
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};

    use crate::gpu::ycbcr::measure::PROBE;
    let (s, t) = rotated_named(p, d, drawn.side(), rotation);
    let v = t.new_texture_view(view).unwrap().unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function("corner_vertex"), function("gradient_fragment"));
    let state = d
        .new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: view,
            resource_index: None,
        })
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&v)),
        load_action: Some(load),
        store_action: Some(1),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    let set = RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(&state),
    };
    let draw = RenderCmd::DrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
    };
    let cmds = match drawn {
        Drawn::Nothing => vec![set],
        Drawn::Corner | Drawn::FirstTile => vec![set, draw],
    };
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    surface_bytes(&s)
}

/// The surface's bytes after a pass that clears a 4-sample texture of `format` and resolves
/// it into a texture of `rotation` ([`rotated_named`], 32 by 32), or into its view in
/// `format` where that is not the texture's, on `p`'s device `d`.
fn rotated_resolve<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    rotation: u64,
    format: u64,
) -> Vec<u8> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::types::ClearColor;
    use reims_vgpu_core::platform::values::ColorAttachmentSet;
    let (s, t) = rotated_named(p, d, 32, rotation);
    let v = (format != 80).then(|| t.new_texture_view(format).unwrap().unwrap());
    let msaa = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: 4,
                sample_count: 4,
                pixel_format: format,
                width: 32,
                height: 32,
                usage: 4,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&msaa)),
        resolve_texture: Some(Some(v.as_ref().unwrap_or(&t))),
        load_action: Some(CLEAR),
        store_action: Some(2),
        clear_color: Some(ClearColor {
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 1.0,
        }),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    surface_bytes(&s)
}

/// A pass through a view of a texture of each rotation over an IOSurface, in
/// `BGRA8Unorm_sRGB`, `RGBA8Unorm` or `RGBA8Unorm_sRGB`, stores and loads the surface's bytes
/// as Metal's does, byte for byte: one that clears or loads and draws over every 32 by 32 tile
/// of its frame, and one that clears or loads and draws nothing. One that loads a texture of
/// rotation 4 and draws dies at `endEncoding` naming `rotated-load-tiles`
/// ([`crate::answer::DIES`]), which the runtime announces at every such pass, though this one's
/// draw reaches every tile; Metal's side of it runs
#[test]
fn views_of_rotated_textures_store_and_load_as_metals() {
    let (p, d) = (platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let passes = [CLEAR, LOAD].map(|l| [(l, Drawn::Corner), (l, Drawn::Nothing)]);
    for view in [81, 70, 71] {
        for pass in passes.into_iter().flatten() {
            for rotation in 1..=4 {
                type V = crate::platform::VulkanPlatform;
                let theirs = rotated_view_pass::<MetalPlatform>(&mp, &md, rotation, view, pass);
                if (rotation, pass) == (4, (LOAD, Drawn::Corner)) {
                    let kind = rt::Degradation::RotatedLoadTiles;
                    let site = crate::answer::dies_of(kind, || {
                        rotated_view_pass::<V>(&p, &d, rotation, view, pass)
                    });
                    assert_eq!(site.file, "endEncoding", "view {view}, rotation {rotation}");
                    continue;
                }
                let ours = rotated_view_pass::<V>(&p, &d, rotation, view, pass);
                let differ = ours.chunks(4).zip(theirs.chunks(4));
                let differ = differ.filter(|(a, b)| a != b).count();
                let case = format!("view {view}, rotation {rotation}, {pass:?}");
                assert_eq!(differ, 0, "{case}");
            }
        }
    }
}

/// A pass that resolves a cleared 4-sample texture into a texture of each rotation over an
/// IOSurface, or into its view in `BGRA8Unorm_sRGB` or `RGBA8Unorm`, stores the surface's
/// bytes as Metal's does, byte for byte
#[test]
fn rotated_textures_and_their_views_resolve_as_metals() {
    let (p, d) = (platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    for format in [80, 81, 70] {
        for rotation in 1..=4 {
            type V = crate::platform::VulkanPlatform;
            let ours = rotated_resolve::<V>(&p, &d, rotation, format);
            let theirs = rotated_resolve::<MetalPlatform>(&mp, &md, rotation, format);
            let differ = ours.chunks(4).zip(theirs.chunks(4));
            let differ = differ.filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "format {format}, rotation {rotation}");
        }
    }
}

/// A view of a texture of each rotation over an IOSurface in a format other than the four
/// (`R32Uint`), or of another type (a 2D array), dies at its selector, where Metal makes it.
/// No view of another range can be asked: the texture has one level and one slice
#[test]
fn views_of_rotated_textures_the_runtime_refuses_die() {
    use reims_vgpu_core::platform::resource::Texture as _;
    let (p, d) = (platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let one = NsRange::new(0, 1);
    for rotation in 1..=4 {
        let (_s, t) = rotated_named(&p, &d, 32, rotation);
        let site = catch_death(|| t.new_texture_view(53).map(drop)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newTextureViewWithPixelFormat:"),
            "rotation {rotation}"
        );
        let site = catch_death(|| t.new_texture_subview(80, 3, one, one).map(drop)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                DeathKind::Abort,
                "newTextureViewWithPixelFormat:textureType:levels:slices:"
            ),
            "rotation {rotation}"
        );
        let Some(md) = &md else { continue };
        let (_s, mt) = rotated_named(&mp, md, 32, rotation);
        assert!(
            mt.new_texture_view(53).unwrap().is_some(),
            "Metal: rotation {rotation}"
        );
        let typed = mt.new_texture_subview(80, 3, one, one).unwrap();
        assert!(typed.is_some(), "Metal: rotation {rotation}");
    }
}

/// `MTLLoadActionLoad`, `MTLLoadActionClear`.
const LOAD: u64 = 1;

const CLEAR: u64 = 2;

/// A pass that loads a texture of rotation 4 and draws over some of its frame's 32 by 32 tiles
/// dies at `endEncoding` naming `rotated-load-tiles` ([`crate::answer::DIES`]): the runtime
/// lands the tiles no draw reaches mirrored as the others, where Metal loads and stores the
/// tiles a draw reaches alone, keeping the others in place. Metal's side runs first
#[test]
fn rotated_loads_of_some_tiles_are_metals() {
    let (p, d) = (platform(), dev());
    let pass = (LOAD, Drawn::FirstTile);
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
        rotated_uses::<MetalPlatform>(&mp, &md, 4, pass);
    }
    let kind = rt::Degradation::RotatedLoadTiles;
    let site = crate::answer::dies_of(kind, || {
        rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, 4, pass)
    });
    assert_eq!(site.file, "endEncoding");
}

/// Textures of format 84 (plane 1's view format of the two-plane XR10 textures) and of the
/// XR10 formats are made with read and write usage, as Metal makes them on this host, which
/// answers supportsExtendedXR10Formats yes
#[test]
fn xr10_textures_are_made() {
    let formats = [84, 550, 551, 552, 553, 554, 555];
    let made_on = |made: &dyn Fn(u64) -> Option<u64>| formats.map(made);
    let d = dev();
    let ours = made_on(&|format| {
        let desc = bgra(8, 8, 0);
        desc.set_usage(0x3);
        desc.set_pixel_format(format);
        let t = d.new_texture(OnFault::Return, &desc).unwrap();
        t.map(|t| t.pixel_format())
    });
    assert_eq!(ours, formats.map(Some));
    assert!(d.caps().flags.supports_extended_xr10_formats);
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let theirs = made_on(&|format| {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
            pixel_format: format,
            width: 8,
            height: 8,
            usage: 0x3,
            ..TextureDesc::default()
        });
        let t = metal.new_texture(OnFault::Return, &desc).unwrap();
        t.map(|t| t.pixel_format())
    });
    assert_eq!(ours, theirs);
    assert!(metal.caps().flags.supports_extended_xr10_formats);
}

/// What 2D samples of a 24 by 12 texture of `format` (550 or 551) read on platform `P`, as
/// halves: its texels written by a kernel from an `RGBA16Float` texture of `seed`'s halves of
/// every sign, below 2 in magnitude, then sampled at positions a third of a texel apart from
/// two texels before the texture to two past it, nearest and linear, in texel coordinates
/// clamped to the edge and in normalized ones repeated.
fn two_plane_samples_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    format: u64,
    seed: u32,
) -> Vec<Vec<u8>> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::{Library as _, Texture as _};

    use crate::gpu::ycbcr::measure::{PROBE, probed};
    use crate::gpu::ycbcr::measure::{probe, sampler};
    let (w, h) = (24, 12);
    let texture = |pixel_format, usage| {
        d.new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format,
                width: w,
                height: h,
                usage,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap()
    };
    let (source, t) = (texture(115, 0x1), texture(format, 0x3));
    let mut x = seed;
    let halves: Vec<u8> = (0..w * h * 4)
        .flat_map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // A sign, an exponent field below 16 (a magnitude below 2) and ten bits of
            // mantissa.
            let bits = (x >> 16) & 0xbfff;
            u16::try_from(bits).unwrap().to_le_bytes()
        })
        .collect();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    source
        .replace_region(all, 0, 0, HostRange::of_slice(&halves), w * 8, 0)
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "copy_texels")
        .unwrap()
        .unwrap();
    let copy = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&copy) },
        ComputeCmd::SetTextures {
            textures: &[Some(&source), Some(&t)],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(w, h, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let grid = (w * 3 + 12, h * 3 + 12);
    let mut reads = Vec::new();
    for filter in [0, 1] {
        for (address, normalized) in [(0, false), (2, true)] {
            let s = sampler::<P>(d, (filter, address, normalized), 0);
            let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, normalized, 0);
            let kernel = "sample_only_texels";
            reads.push(probed::<P>(d, kernel, &t, grid, Some(&s), Some(&at)));
        }
    }
    reads
}

/// A texture of a two-plane XR10 format, 550 or 551, written by a kernel, samples in 2D as the
/// Metal platform's does on this host, bit for bit: nearest and linear, clamped to the edge
/// and repeated
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
fn two_plane_xr10_samples_are_the_metal_platforms() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let Some(metal) = MetalPlatform::system_default_device() else {
            return;
        };
        let d = dev();
        let vulkan_desc = |t: &TextureDesc| platform().texture_descriptor(t);
        let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
        for format in [550, 551] {
            for seed in [3, 4] {
                let ours = two_plane_samples_on::<VulkanPlatform>(&d, &vulkan_desc, format, seed);
                let theirs =
                    two_plane_samples_on::<MetalPlatform>(&metal, &metal_desc, format, seed);
                for (i, read) in ours.iter().enumerate() {
                    let values: std::collections::HashSet<&[u8]> = read.chunks(8).collect();
                    assert!(
                        values.len() > 100,
                        "{format} seed {seed} case {i}: {}",
                        values.len()
                    );
                }
                for (i, (o, m)) in ours.iter().zip(&theirs).enumerate() {
                    let differ = o.chunks(8).zip(m.chunks(8)).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differ,
                        0,
                        "{format} seed {seed} case {i}: of {}",
                        o.len() / 8
                    );
                }
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// Making a texture of MTLPixelFormatInvalid, or of an empty size, dies at the creator, as
/// Metal aborts there; the descriptor takes either. One of X32_Stencil8 is made, as Metal makes
/// it on this host
#[test]
fn textures_metal_aborts_making_die() {
    let d = dev();
    let sel = "newTextureWithDescriptor:";
    let invalid = bgra(8, 8, 0);
    invalid.set_pixel_format(0);
    let site = catch_death(|| d.new_texture(OnFault::Return, &invalid)).unwrap_err();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    for (width, height) in [(0, 8), (8, 0)] {
        let empty = bgra(width, height, 0);
        let site = catch_death(|| d.new_texture(OnFault::Return, &empty)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }
    let s = bgra_surface(8, 8);
    let site = catch_death(|| d.new_texture_on_surface(&bgra(0, 8, 0), Some(&s), 0)).unwrap_err();
    assert_eq!(site.file, "newTextureWithDescriptor:iosurface:plane:");
    let stencil_view_format = bgra(8, 8, 0x20);
    stencil_view_format.set_pixel_format(261);
    let stencil = d
        .new_texture(OnFault::Return, &stencil_view_format)
        .unwrap();
    assert_eq!(stencil.map(|t| t.pixel_format()), Some(261));
    if let Some(metal) = MetalPlatform::system_default_device() {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
            pixel_format: 261,
            width: 8,
            height: 8,
            usage: 0x13,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        let made = metal.new_texture(OnFault::Return, &desc).unwrap();
        assert_eq!(made.map(|t| t.pixel_format()), Some(261));
    }
}

/// newHeapWithDescriptor: of core's values: the descriptor's options, the heap's and its
/// storage mode are Metal's on this host; a heap of size 0 is nil, as Metal's; a Managed
/// heap dies, as Metal asserts at it
#[test]
fn heaps_of_descriptors() {
    use OptionSet::{HazardTrackingMode as Tracking, StorageMode as Storage};
    let values = [
        vec![],
        vec![Storage(2), Tracking(0)],
        vec![Storage(2), Tracking(2)],
        vec![Storage(2), Tracking(1)],
        vec![Storage(0), Tracking(0)],
    ];
    fn answers<P: Platform>(made: HeapMade<P>) -> (u64, u64, u64) {
        let h = made.heap.unwrap();
        (
            made.descriptor_resource_options,
            h.resource_options(),
            h.storage_mode(),
        )
    }
    let d = dev();
    let ours: Vec<_> = values
        .iter()
        .map(|v| answers(heap_of::<VulkanPlatform>(&d, 0x8000, v)))
        .collect();
    let measured = [
        (0x20, 0x120, 2),
        (0x20, 0x120, 2),
        (0x220, 0x220, 2),
        (0x120, 0x120, 2),
        (0, 0x100, 0),
    ];
    assert_eq!(ours, measured);
    let (a, b) = (private_heap(), private_heap());
    assert!(a.gpu_address() != 0 && a.gpu_address() != b.gpu_address());
    assert!(
        heap_of::<VulkanPlatform>(&d, 0, &[Storage(2)])
            .heap
            .is_none()
    );
    let site = catch_death(|| {
        heap_of::<VulkanPlatform>(&d, 0x8000, &[Storage(1)])
            .heap
            .is_some()
    })
    .unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newHeapWithDescriptor:")
    );
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let theirs: Vec<_> = values
        .iter()
        .map(|v| answers(heap_of::<MetalPlatform>(&metal, 0x8000, v)))
        .collect();
    assert_eq!(theirs, measured);
    assert!(
        heap_of::<MetalPlatform>(&metal, 0, &[Storage(2)])
            .heap
            .is_none()
    );
}

/// What a buffer placed over the bytes of a 16 by 12 texture of `format` (550 or 551), one
/// level, at offset 0 of a placement heap of storage mode `storage` (0 Shared, 2 Private) on
/// `P`'s device `d` sees and changes of the texture's texels: the bytes of the texture's span,
/// each first its offset's low byte, after a kernel writes the texels from an `RGBA16Float`
/// texture of halves of every sign, read through a copy of the buffer; then the texels
/// `read()` reads after another pattern is copied into the buffer. Every read and write is
/// the GPU's, a command buffer each, so a Private heap's bytes are read as a Shared one's.
fn two_plane_heap_bytes_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    format: u64,
    storage: u64,
) -> (Vec<u8>, Vec<u8>)
where
    P::Device: GpuDevice<P>,
{
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Library as _};

    use crate::gpu::ycbcr::measure::{PROBE, probed_texels};
    let (w, h) = (16, 12);
    let options = [
        OptionSet::StorageMode(storage),
        OptionSet::HazardTrackingMode(0),
    ];
    let heap = heap_of::<P>(d, 0x10000, &options).heap.unwrap();
    let texture_desc = desc(&TextureDesc {
        pixel_format: format,
        width: w,
        height: h,
        usage: 0x3,
        resource_options: Some(0x100 | storage << 4),
        ..TextureDesc::default()
    });
    let size = d.heap_texture_size_and_align(&texture_desc).unwrap().size;
    let t = heap.new_texture(&texture_desc, 0).unwrap().unwrap();
    let over = heap
        .new_buffer_of_length(size, storage << 4, 0)
        .unwrap()
        .unwrap();
    let shared = |bytes: &[u8]| {
        let length = u64::try_from(bytes.len()).unwrap();
        let b = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of `bytes.len()` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), bytes.len());
        }
        b
    };
    let q = d.new_command_queue().unwrap();
    let copied = |from: &P::Buffer, to: &P::Buffer| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(from),
            source_offset: 0,
            destination_buffer: Some(to),
            destination_offset: 0,
            size,
        };
        e.encode(OnFault::Die, copy).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    let span = usize::try_from(size).unwrap();
    let offsets: Vec<u8> = (0..span).map(|i| (i & 0xff) as u8).collect();
    copied(&shared(&offsets), &over);
    let source = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: 115,
                width: w,
                height: h,
                usage: 0x1,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let mut x = 5u32;
    let halves: Vec<u8> = (0..w * h * 4)
        .flat_map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u16::try_from((x >> 16) & 0xbfff).unwrap().to_le_bytes()
        })
        .collect();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    source
        .replace_region(all, 0, 0, HostRange::of_slice(&halves), w * 8, 0)
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "copy_texels")
        .unwrap()
        .unwrap();
    let copy = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&copy) },
        ComputeCmd::SetTextures {
            textures: &[Some(&source), Some(&t)],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(w, h, 1),
            threads_per_threadgroup: MtlSize::new(8, 4, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let back = shared(&vec![0; span]);
    copied(&over, &back);
    // SAFETY: a Shared buffer's `span` bytes, read after the copy completed.
    let seen = unsafe { std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), span) };
    let seen = seen.to_vec();
    let pattern: Vec<u8> = (0..span).map(|i| (i * 7 + 3) as u8).collect();
    copied(&shared(&pattern), &over);
    let texels = probed_texels::<P>(d, ("read_texels", 8), &t, (w, h), None, None);
    (seen, texels)
}

/// A buffer placed over a one-level 2D texture of a two-plane XR10 format (550, 551) in a
/// placement heap, Shared or Private, sees and changes the texture's texels as Metal's does
/// on this host, byte for byte, the bytes between the rows and the planes included
/// ([`two_plane_heap_bytes_on`]): the runtime keeps the texture's planes in the heap's bytes,
/// where Metal lays them out, so no placement over it dies ([`Placements`])
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
fn a_buffer_over_a_one_level_two_plane_heap_texture_sees_its_texels_as_metals() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let Some(metal) = MetalPlatform::system_default_device() else {
            return;
        };
        let d = dev();
        let vulkan_desc = |t: &TextureDesc| platform().texture_descriptor(t);
        let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
        for format in [550, 551] {
            for storage in [0, 2] {
                let case = format!("{format} storage {storage}");
                let (seen, texels) =
                    two_plane_heap_bytes_on::<VulkanPlatform>(&d, &vulkan_desc, format, storage);
                let (metal_seen, metal_texels) =
                    two_plane_heap_bytes_on::<MetalPlatform>(&metal, &metal_desc, format, storage);
                let offsets = seen
                    .iter()
                    .enumerate()
                    .filter(|&(i, &b)| b == (i & 0xff) as u8);
                assert!(offsets.count() < seen.len(), "the texels land: {case}");
                let differ = |a: &[u8], b: &[u8], n| {
                    a.chunks(n).zip(b.chunks(n)).filter(|(a, b)| a != b).count()
                };
                assert_eq!(seen.len(), metal_seen.len(), "{case}");
                assert_eq!(differ(&seen, &metal_seen, 4), 0, "seen: {case}");
                assert_eq!(differ(&texels, &metal_texels, 8), 0, "read: {case}");
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// A texture the runtime does not tile placed in a heap over the guest's pages is made, as
/// Metal makes it in the same heap, and the runtime's announcement
/// (`heap-texture-untiled-layout`) is relayed, not a death: a two-aspect
/// `Depth32Float_Stencil8` render target, as the suite's depth and stencil cases place it
/// (core's detach heap, `Shared` over the pages, and the texture of the heap's options, as
/// core sets them whatever the guest's descriptor names)
#[test]
fn heap_textures_over_guest_pages_in_an_untiled_layout_are_made() {
    use crate::os::OsVm;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::types::ResourceAddressRange;

    const SIZE: u64 = 0x10000;

    /// Whether a `Depth32Float_Stencil8` render target is made at 0 in a `Shared` heap of
    /// `SIZE` bytes over pages of their own, of the heap's options.
    fn placed<P: Platform>(p: &P, d: &P::Device) -> bool {
        // Left mapped: the heap's alias shares them until the process ends.
        let pages = OsVm::new().allocate_anywhere(SIZE).unwrap();
        let ranges = p
            .new_address_range_array(&[ResourceAddressRange {
                address: u64::try_from(pages.addr()).unwrap(),
                length: SIZE,
            }])
            .unwrap();
        let heap = d
            .new_heap(&HeapDesc {
                size: SIZE,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let depth_stencil = p.texture_descriptor(&TextureDesc {
            pixel_format: 260,
            width: 16,
            height: 16,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            ..TextureDesc::default()
        });
        heap.new_texture(&depth_stencil, 0).unwrap().is_some()
    }

    let (p, d) = (platform(), dev());
    crate::answer::RELAYED.lock().clear();
    assert!(placed(&p, &d));
    let relayed = crate::answer::RELAYED.lock().clone();
    let untiled = rt::Degradation::HeapTextureUntiledLayout;
    assert!(relayed.contains(&untiled), "{relayed:?}");
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        assert!(placed(&mp, &md));
    }
}

/// A `Shared` no-copy buffer of [`over_pages`] bytes over pages of its own, detached at once,
/// as core's detach buffer is made: both platforms refuse a replace of an attached buffer.
fn detached_buffer<P: Platform>(d: &P::Device) -> P::Buffer {
    let own = usize::try_from(guest_pages()).unwrap();
    // SAFETY: the pages stay mapped until the process ends.
    let contents = unsafe { HostRange::new(HostPtr::from_addr(own), over_pages()) };
    let buffer = d
        .new_buffer(BufferDesc {
            length: over_pages(),
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: Some(contents),
            no_copy: true,
            deallocator: None,
        })
        .unwrap()
        .unwrap();
    buffer.detach_backing().unwrap();
    buffer
}

/// `buffer`'s backing replaced with the pages at `address`, as core's first prepare gives a
/// detached buffer the guest's pages.
fn replace_buffer<P: Platform>(p: &P, buffer: &P::Buffer, address: u64) {
    let replaced = buffer.replace_backing(&ranges_at(p, address), false);
    assert!(replaced.unwrap(), "the buffer's backing replaced");
}

/// A buffer over the pages at `address`, made as core makes its detach buffer: a
/// [`detached_buffer`], given the pages by [`replace_buffer`].
fn buffer_at<P: Platform>(p: &P, d: &P::Device, address: u64) -> P::Buffer {
    let buffer = detached_buffer::<P>(d);
    replace_buffer(p, &buffer, address);
    buffer
}

/// A surface of one page of `BGRA8Unorm` texels, 16 by 16, made over the guest's page at
/// `address`, as core makes its surfaces over the guest's memory, and a texture descriptor
/// for a texture on it.
fn surface_at<P: Platform<Surface = OsSurface>>(
    p: &P,
    address: u64,
) -> (OsSurface, P::TextureDescriptor) {
    use reims_vgpu_core::host::PhysicalMemoryRange;
    let surface = p.surfaces().create(&SurfaceSpec {
        pixel_format: u32::from_be_bytes(*b"BGRA"),
        alloc_size: 0x4000,
        width: 16,
        height: 16,
        bytes_per_element: 4,
        bytes_per_row: 64,
        element_width: 1,
        element_height: 1,
        client_address_ranges: Some(vec![PhysicalMemoryRange {
            physical_address: address,
            physical_length: 0x4000,
        }]),
        ..SurfaceSpec::default()
    });
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: 16,
        height: 16,
        usage: 0x1,
        ..TextureDesc::default()
    });
    (surface.unwrap(), desc)
}

/// An 8x8 `BGRA8Unorm` texture of `options` over a buffer's bytes, rows 256 bytes apart.
fn linear_of<P: Platform>(p: &P, options: u64) -> P::TextureDescriptor {
    p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: 8,
        height: 8,
        usage: 0x1,
        resource_options: Some(options),
        ..TextureDesc::default()
    })
}

/// A fill of the bytes `range` of `buffer`, committed and waited for.
fn fill_range<P: Platform>(d: &P::Device, buffer: &P::Buffer, range: NsRange) {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let fill = BlitCmd::FillBuffer {
        buffer,
        range,
        value: 0x5c,
    };
    e.encode(OnFault::Return, fill).unwrap();
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// A compute pass that makes `heap` resident (`useHeaps:count:`), committed and waited for.
fn use_heap<P: Platform>(d: &P::Device, heap: &P::Heap) {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
    let heaps = [Some(heap)];
    e.encode(OnFault::Return, ComputeCmd::UseHeaps { heaps: &heaps })
        .unwrap();
    e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// A buffer or texture placed over a byte of a texture of a heap over the guest's pages that
/// the runtime does not lay out in Metal's tiles dies at its placement while the texture
/// lives, and so does such a texture placed over a byte of a buffer that lives: the runtime
/// announced the texture at its creation (`heap-texture-untiled-layout`,
/// [`crate::answer::HEARD`]). A buffer beside it is made, and one over its bytes once it is
/// let go. Metal makes each of them in the same heap: a two-aspect `Depth32Float_Stencil8`
/// render target, as the suite's cases place it, of the heap's options, beside a buffer over
/// all of the heap, as core makes with each heap. Here the heap has no such buffer, beside which
/// the texture dies (`untiled_textures_die_beside_the_heaps_cover`)
#[test]
fn placements_over_untiled_heap_textures_die() {
    use reims_vgpu_core::raise::death::DeathSite;

    const BUFFER: &str = "newBufferWithLength:options:offset:";
    const TEXTURE: &str = "newTextureWithDescriptor:offset:";
    let size = over_pages();

    // Metal's side first: each placement is made.
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let (heap, depth_stencil, beside) = heap_over_pages(&mp, &md);
        let options = heap.resource_options();
        let whole = heap.new_buffer_of_length(size, options, 0).unwrap();
        assert!(whole.is_some(), "Metal: a buffer over all of the heap");
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture");
        let over = heap.new_buffer_of_length(0x100, options, 0).unwrap();
        assert!(over.is_some(), "Metal: a buffer over it");
        let b = heap.new_buffer_of_length(0x100, options, beside).unwrap();
        assert!(b.is_some(), "Metal: a buffer beside it");
        let second = heap.new_texture(&depth_stencil, beside).unwrap();
        assert!(second.is_some(), "Metal: a texture over a buffer");
        drop((t, over, second));
        let after = heap.new_buffer_of_length(0x100, options, 0).unwrap();
        assert!(after.is_some(), "Metal: a buffer once it is let go");
        drop(whole);
    }

    let dies = |site: DeathSite, sel, case: &str| {
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{case}: {line}"
        );
        assert!(
            line.contains("heap-texture-untiled-layout"),
            "{case}: {line}"
        );
    };
    let (p, d) = (platform(), dev());
    let (heap, depth_stencil, beside) = heap_over_pages(&p, &d);
    let options = heap.resource_options();
    crate::answer::RELAYED.lock().clear();
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let untiled = rt::Degradation::HeapTextureUntiledLayout;
    assert!(crate::answer::RELAYED.lock().contains(&untiled));
    *crate::answer::LAST_LINE.lock() = None;
    dies(
        catch_death(|| heap.new_buffer_of_length(0x100, options, 0)).unwrap_err(),
        BUFFER,
        "a buffer over it",
    );
    let b = heap
        .new_buffer_of_length(0x100, options, beside)
        .unwrap()
        .unwrap();
    *crate::answer::LAST_LINE.lock() = None;
    dies(
        catch_death(|| heap.new_texture(&depth_stencil, beside)).unwrap_err(),
        TEXTURE,
        "a texture over a buffer that lives",
    );
    drop(t);
    assert!(
        (heap.new_buffer_of_length(0x100, options, 0).unwrap()).is_some(),
        "a buffer once it is let go"
    );
    drop(b);
}

/// A texture the runtime does not lay out in Metal's tiles (`heap-texture-untiled-layout`)
/// placed in a heap over the guest's pages beside its cover, core's buffer over all of it, made
/// first, as five suite cases place one, dies at its placement, naming the cover and the kind:
/// the cover and the guest's pages would show the runtime's layout where Metal's are its tiles
/// ([`Placements`]). So does one placed after the heap was made resident. Once the cover is let
/// go, the texture is made. Metal makes each, runs a fill of the cover over the texture, and
/// makes the heap resident
#[test]
fn untiled_textures_die_beside_the_heaps_cover() {
    const TEXTURE: &str = "newTextureWithDescriptor:offset:";

    // Metal's side first: each is made, and the fill and the pass run.
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let (heap, depth_stencil, _) = heap_over_pages(&mp, &md);
        let options = heap.resource_options();
        let cover = heap.new_buffer_of_length(over_pages(), options, 0).unwrap();
        let cover = cover.unwrap();
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture");
        fill_range::<MetalPlatform>(&md, &cover, NsRange::new(0, 16));
        drop(t);
        use_heap::<MetalPlatform>(&md, &heap);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture once the heap was resident");
        drop((t, cover));
    }

    let (p, d) = (platform(), dev());
    let dies = |case: &str, place: &dyn Fn()| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(place).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, TEXTURE),
            "{case}: {line}"
        );
        let beside = "while the buffer over all of the heap that core makes with it lives";
        assert!(line.contains(beside), "{case}: {line}");
        assert!(
            line.contains("heap-texture-untiled-layout"),
            "{case}: {line}"
        );
    };
    let (heap, depth_stencil, _) = heap_over_pages(&p, &d);
    let options = heap.resource_options();
    let cover = heap
        .new_buffer_of_length(over_pages(), options, 0)
        .unwrap()
        .unwrap();
    dies("beside the cover", &|| {
        drop(heap.new_texture(&depth_stencil, 0));
    });
    use_heap::<VulkanPlatform>(&d, &heap);
    dies("once the heap was made resident", &|| {
        drop(heap.new_texture(&depth_stencil, 0));
    });
    drop(cover);
    let t = heap.new_texture(&depth_stencil, 0).unwrap();
    assert!(t.is_some(), "the texture once the cover is let go");
}

/// A texture the runtime does not lay out in Metal's tiles (`heap-texture-untiled-layout`)
/// in a heap over the guest's pages, against the objects core may make over the same pages
/// (a buffer whose backing is replaced with them, a second heap over them): while it lives, a
/// fill of the buffer over its pages dies and one beside them runs, a texture over the
/// buffer's bytes over them dies, a texture of the second heap over them dies, and the second
/// heap made resident dies, each naming the kind; the first heap made resident, with no cover,
/// runs; such a texture placed over pages a texture over another buffer lies over dies; once
/// it is let go, the fill over its pages runs. Making the buffer and the second heap is no
/// death. Metal makes and runs each
#[test]
fn objects_over_the_pages_under_an_untiled_texture_die() {
    use reims_vgpu_core::raise::death::DeathSite;
    const FILL: &str = "fillBuffer:range:value:";
    const TEXTURE: &str = "newTextureWithDescriptor:offset:";
    const OVER: &str = "newTextureWithDescriptor:offset:bytesPerRow:";
    const USE_HEAPS: &str = "useHeaps:count:";

    // Metal's side first: each is made, and each pass runs.
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let at = guest_pages();
        let (heap, depth_stencil, beside) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture");
        let b = buffer_at(&mp, &md, at);
        fill_range::<MetalPlatform>(&md, &b, NsRange::new(0, 16));
        fill_range::<MetalPlatform>(&md, &b, NsRange::new(beside, 16));
        let options = b.resource_options();
        let over = b.new_texture(OnFault::Return, &linear_of(&mp, options), 0, 256);
        assert!(over.unwrap().is_some(), "Metal: a texture over the buffer");
        let second = heap_at(&mp, &md, at);
        let placed = second.new_texture(&linear_of(&mp, second.resource_options()), 0);
        assert!(
            placed.unwrap().is_some(),
            "Metal: a texture of the second heap"
        );
        use_heap::<MetalPlatform>(&md, &second);
        use_heap::<MetalPlatform>(&md, &heap);
        let at = guest_pages();
        let b = buffer_at(&mp, &md, at);
        let over = b.new_texture(OnFault::Return, &linear_of(&mp, options), 0, 256);
        assert!(
            over.unwrap().is_some(),
            "Metal: a texture over a buffer first"
        );
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture over its pages");
    }

    let dies = |site: DeathSite, sel, case: &str| {
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{case}: {line}"
        );
        let pages = "bytes over the guest's pages that a texture of a heap over them lies over";
        assert!(line.contains(pages), "{case}: {line}");
        assert!(
            line.contains("heap-texture-untiled-layout"),
            "{case}: {line}"
        );
    };
    let (p, d) = (platform(), dev());
    let at = guest_pages();
    let (heap, depth_stencil, beside) = heap_over_pages_at(&p, &d, at);
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let b = buffer_at(&p, &d, at);
    *crate::answer::LAST_LINE.lock() = None;
    dies(
        catch_death(|| fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16))).unwrap_err(),
        FILL,
        "a fill of the buffer over its pages",
    );
    fill_range::<VulkanPlatform>(&d, &b, NsRange::new(beside, 16));
    let options = b.resource_options();
    dies(
        catch_death(|| b.new_texture(OnFault::Return, &linear_of(&p, options), 0, 256))
            .unwrap_err(),
        OVER,
        "a texture over the buffer's bytes over its pages",
    );
    let second = heap_at(&p, &d, at);
    let linear = linear_of(&p, second.resource_options());
    dies(
        catch_death(|| second.new_texture(&linear, 0)).unwrap_err(),
        TEXTURE,
        "a texture of the second heap over its pages",
    );
    dies(
        catch_death(|| use_heap::<VulkanPlatform>(&d, &second)).unwrap_err(),
        USE_HEAPS,
        "the second heap made resident",
    );
    use_heap::<VulkanPlatform>(&d, &heap);
    let at2 = guest_pages();
    let b2 = buffer_at(&p, &d, at2);
    let over = b2
        .new_texture(OnFault::Return, &linear_of(&p, options), 0, 256)
        .unwrap();
    assert!(over.is_some(), "a texture over a buffer first");
    let (heap2, depth_stencil2, _) = heap_over_pages_at(&p, &d, at2);
    dies(
        catch_death(|| heap2.new_texture(&depth_stencil2, 0)).unwrap_err(),
        TEXTURE,
        "the texture over pages a texture over a buffer lies over",
    );
    drop(t);
    fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16));
    drop((over, b2, b, second));
}

/// The death line of a death over the guest's pages ([`crate::gpu::over_pages`]) at `sel`,
/// whose message holds `text`, of an untiled texture of a heap.
fn dies_over_pages(site: reims_vgpu_core::raise::death::DeathSite, sel: &str, text: &str) {
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, sel),
        "{text}: {line}"
    );
    assert!(line.contains(text), "{text}: {line}");
    assert!(line.contains("heap-texture-untiled-layout"), "{line}");
}

/// The text of a death of objects over the guest's pages under an untiled texture.
const UNDER_PAGES: &str =
    "bytes over the guest's pages that a texture of a heap over them lies over";

/// Core's flow: a heap or buffer made over pages of its own and given the guest's pages by a
/// replace at its first prepare, after the textures in or over it are made. A replace that
/// brings a heap's untiled texture (`heap-texture-untiled-layout`) over pages a texture over a
/// buffer lies over dies, and so do one that brings a buffer's texture under such a texture
/// and one that brings a second heap's texture under it: a use of either texture would meet
/// the other's bytes where Metal's meets its own. Metal makes each, and each replace answers
/// YES
#[test]
fn replaces_that_bring_objects_over_an_untiled_textures_pages_die() {
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let at = guest_pages();
        let heap = detached_heap(&mp, &md);
        let (depth_stencil, _) = depth_stencil_of(&mp, &md, &heap);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture before its heap's replace");
        let b = buffer_at(&mp, &md, at);
        let linear = linear_of(&mp, b.resource_options());
        let over = b.new_texture(OnFault::Return, &linear, 0, 256).unwrap();
        assert!(over.is_some(), "Metal: a texture over the buffer");
        replace_heap(&mp, &heap, at);
        let at = guest_pages();
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture");
        let b = detached_buffer::<MetalPlatform>(&md);
        let over = b.new_texture(OnFault::Return, &linear, 0, 256).unwrap();
        assert!(over.is_some(), "Metal: a texture over a detached buffer");
        replace_buffer(&mp, &b, at);
        let second = detached_heap(&mp, &md);
        let linear = linear_of(&mp, second.resource_options());
        let placed = second.new_texture(&linear, 0).unwrap();
        assert!(placed.is_some(), "Metal: a texture of a detached heap");
        replace_heap(&mp, &second, at);
    }

    let (p, d) = (platform(), dev());
    let at = guest_pages();
    let heap = detached_heap(&p, &d);
    let (depth_stencil, _) = depth_stencil_of(&p, &d, &heap);
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let b = buffer_at(&p, &d, at);
    let linear = linear_of(&p, b.resource_options());
    let over = b.new_texture(OnFault::Return, &linear, 0, 256).unwrap();
    let over = over.unwrap();
    *crate::answer::LAST_LINE.lock() = None;
    dies_over_pages(
        catch_death(|| replace_heap(&p, &heap, at)).unwrap_err(),
        REPLACE,
        UNDER_PAGES,
    );
    drop((t, over));
    let at = guest_pages();
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let b = detached_buffer::<VulkanPlatform>(&d);
    let over = b.new_texture(OnFault::Return, &linear, 0, 256).unwrap();
    let over = over.unwrap();
    dies_over_pages(
        catch_death(|| replace_buffer(&p, &b, at)).unwrap_err(),
        REPLACE,
        UNDER_PAGES,
    );
    drop(over);
    let second = detached_heap(&p, &d);
    let linear = linear_of(&p, second.resource_options());
    let placed = second.new_texture(&linear, 0).unwrap().unwrap();
    dies_over_pages(
        catch_death(|| replace_heap(&p, &second, at)).unwrap_err(),
        REPLACE,
        UNDER_PAGES,
    );
    drop((placed, t));
}

/// A command's use of a buffer's bytes over the guest's pages, run before any texture is
/// kept apart, against an untiled texture of a heap over the same pages: the texture placed
/// in a heap over them dies, and so does the replace that brings a heap's such texture over
/// them, each naming the use; once the buffer is detached, which gives its pages up, the
/// texture is placed. Metal makes and runs each
#[test]
fn a_texture_kept_apart_over_pages_a_command_used_dies() {
    const TEXTURE: &str = "newTextureWithDescriptor:offset:";
    const USED: &str = "that a command used through another heap or a buffer";
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let at = guest_pages();
        let b = buffer_at(&mp, &md, at);
        fill_range::<MetalPlatform>(&md, &b, NsRange::new(0, 16));
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture over the used pages");
        let detached = detached_heap(&mp, &md);
        let (depth_stencil, _) = depth_stencil_of(&mp, &md, &detached);
        let t = detached.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture before its heap's replace");
        replace_heap(&mp, &detached, at);
    }

    let (p, d) = (platform(), dev());
    let at = guest_pages();
    let b = buffer_at(&p, &d, at);
    fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16));
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    *crate::answer::LAST_LINE.lock() = None;
    dies_over_pages(
        catch_death(|| heap.new_texture(&depth_stencil, 0)).unwrap_err(),
        TEXTURE,
        USED,
    );
    let detached = detached_heap(&p, &d);
    let (detached_depth_stencil, _) = depth_stencil_of(&p, &d, &detached);
    let t = detached.new_texture(&detached_depth_stencil, 0).unwrap();
    let t = t.unwrap();
    dies_over_pages(
        catch_death(|| replace_heap(&p, &detached, at)).unwrap_err(),
        REPLACE,
        USED,
    );
    drop(t);
    b.detach_backing().unwrap();
    let placed = heap.new_texture(&depth_stencil, 0).unwrap();
    assert!(placed.is_some(), "placed once the buffer gave its pages up");
}

/// A texture on a surface made over the guest's pages, as core makes its surfaces over the
/// guest's memory, against an untiled texture of a heap over the same pages: made after the
/// heap's texture it dies, and made first, the heap's texture placed over them dies. Metal
/// makes each
#[test]
fn textures_on_a_surface_over_an_untiled_textures_pages_die() {
    const ON: &str = "newTextureWithDescriptor:iosurface:plane:";
    const TEXTURE: &str = "newTextureWithDescriptor:offset:";
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let at = guest_pages();
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the heap's texture");
        let (surface, bgra) = surface_at(&mp, at);
        let on = md.new_texture_on_surface(&bgra, Some(&surface), 0).unwrap();
        assert!(on.is_some(), "Metal: the texture on the surface");
        let at = guest_pages();
        let (surface, bgra) = surface_at(&mp, at);
        let on = md.new_texture_on_surface(&bgra, Some(&surface), 0).unwrap();
        assert!(on.is_some(), "Metal: the texture on the surface first");
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the heap's texture over its pages");
    }

    let (p, d) = (platform(), dev());
    let at = guest_pages();
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let (surface, bgra) = surface_at(&p, at);
    *crate::answer::LAST_LINE.lock() = None;
    dies_over_pages(
        catch_death(|| d.new_texture_on_surface(&bgra, Some(&surface), 0)).unwrap_err(),
        ON,
        UNDER_PAGES,
    );
    drop(t);
    let at = guest_pages();
    let (surface, bgra) = surface_at(&p, at);
    let on = d.new_texture_on_surface(&bgra, Some(&surface), 0).unwrap();
    let on = on.unwrap();
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    dies_over_pages(
        catch_death(|| heap.new_texture(&depth_stencil, 0)).unwrap_err(),
        TEXTURE,
        UNDER_PAGES,
    );
    drop(on);
}

/// A no-copy buffer over a mapping of the guest's pages of the platform's own, as core makes
/// a task's transfer buffers (`new_buffer_for_virtual_page`: a fresh remap through the
/// platform's Mach VM), against an untiled texture (`heap-texture-untiled-layout`) of a heap
/// over the same pages, which name them by other host addresses: a fill of it over the
/// texture's bytes dies, and so does a texture over its bytes there, each naming the kind;
/// a fill of such a buffer made first, over other pages, kills the placement of such a
/// texture over them, naming the use ([`crate::remap::as_guests`]). Metal makes and runs each
#[test]
fn buffers_over_a_mapping_of_the_guests_pages_meet_untiled_textures() {
    const FILL: &str = "fillBuffer:range:value:";
    const OVER: &str = "newTextureWithDescriptor:offset:bytesPerRow:";
    const TEXTURE: &str = "newTextureWithDescriptor:offset:";
    const USED: &str = "that a command used through another heap or a buffer";
    /// A no-copy buffer over a fresh remap of the [`over_pages`] bytes at `at`, left mapped.
    fn mapped<P: Platform>(p: &P, d: &P::Device, at: u64) -> P::Buffer {
        let vm = p.vm();
        let mapping = vm.allocate_anywhere(over_pages()).unwrap();
        let source = HostPtr::from_addr(usize::try_from(at).unwrap());
        vm.remap_shared_overwrite(mapping, source, over_pages())
            .unwrap();
        // SAFETY: the mapping stays mapped until the process ends.
        let bytes = unsafe { HostRange::new(mapping, over_pages()) };
        let made = d.new_buffer_over_host_memory(OnFault::Return, bytes, 0, None);
        made.unwrap().unwrap()
    }
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let at = guest_pages();
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture");
        let b = mapped(&mp, &md, at);
        fill_range::<MetalPlatform>(&md, &b, NsRange::new(0, 16));
        let linear = linear_of(&mp, b.resource_options());
        let over = b.new_texture(OnFault::Return, &linear, 0, 256).unwrap();
        assert!(over.is_some(), "Metal: a texture over the mapping");
        let at = guest_pages();
        let b = mapped(&mp, &md, at);
        fill_range::<MetalPlatform>(&md, &b, NsRange::new(0, 16));
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "Metal: the texture over the used pages");
    }

    let (p, d) = (platform(), dev());
    let at = guest_pages();
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let b = mapped(&p, &d, at);
    *crate::answer::LAST_LINE.lock() = None;
    dies_over_pages(
        catch_death(|| fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16))).unwrap_err(),
        FILL,
        UNDER_PAGES,
    );
    let linear = linear_of(&p, b.resource_options());
    dies_over_pages(
        catch_death(|| b.new_texture(OnFault::Return, &linear, 0, 256)).unwrap_err(),
        OVER,
        UNDER_PAGES,
    );
    drop(t);
    let at = guest_pages();
    let b = mapped(&p, &d, at);
    fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16));
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    dies_over_pages(
        catch_death(|| heap.new_texture(&depth_stencil, 0)).unwrap_err(),
        TEXTURE,
        USED,
    );
}

/// A view of a texture the runtime does not lay out in Metal's tiles
/// (`heap-texture-untiled-layout`), in a heap over the guest's pages, keeps the texture's
/// bytes taken once the texture is released, as a Metal view's `parentTexture` keeps the
/// texture ([`TextureObject::life`]): a buffer placed over them in the heap dies, and so does
/// a fill of a buffer over the guest's pages under them, each naming the kind; once the view
/// is let go too, the buffer is made and the fill runs. Metal makes and runs each
#[test]
fn a_view_keeps_its_released_textures_bytes() {
    const BUFFER: &str = "newBufferWithLength:options:offset:";
    const FILL: &str = "fillBuffer:range:value:";
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let at = guest_pages();
        let (heap, depth_stencil, _) = heap_over_pages_at(&mp, &md, at);
        let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
        let view = t.new_texture_view(260).unwrap();
        assert!(view.is_some(), "Metal: the view");
        drop(t);
        let over = heap.new_buffer_of_length(0x100, heap.resource_options(), 0);
        assert!(over.unwrap().is_some(), "Metal: a buffer over its bytes");
        let b = buffer_at(&mp, &md, at);
        fill_range::<MetalPlatform>(&md, &b, NsRange::new(0, 16));
        drop(view);
    }

    let (p, d) = (platform(), dev());
    let at = guest_pages();
    let (heap, depth_stencil, _) = heap_over_pages_at(&p, &d, at);
    let options = heap.resource_options();
    let t = heap.new_texture(&depth_stencil, 0).unwrap().unwrap();
    let view = t.new_texture_view(260).unwrap().unwrap();
    drop(t);
    let b = buffer_at(&p, &d, at);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| heap.new_buffer_of_length(0x100, options, 0)).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, BUFFER), "{line}");
    assert!(line.contains("heap-texture-untiled-layout"), "{line}");
    dies_over_pages(
        catch_death(|| fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16))).unwrap_err(),
        FILL,
        UNDER_PAGES,
    );
    drop(view);
    let over = heap.new_buffer_of_length(0x100, options, 0).unwrap();
    assert!(over.is_some(), "a buffer over them once the view is let go");
    fill_range::<VulkanPlatform>(&d, &b, NsRange::new(0, 16));
}

/// A 4-sample `RGBA8Unorm` render target placed in a `Shared` heap over the guest's pages, of
/// the heap's options as core gives them, dies at the placement. Where Metal keeps it
/// losslessly compressed (16 by 16, its sample grid 32 by 32, with
/// `allowGPUOptimizedContents`), the runtime makes it, its image in memory of its own, the
/// heap's bytes keeping what they held, and announces `heap-texture-lossless-layout`, which the
/// placement hears and dies of, naming it. One Metal keeps uncompressed,
/// whose samples Metal lays out in the heap's bytes, dies naming the runtime's refusal
/// (`NotImplemented(HostHeapUncompressedMultisample)`): 16 by 16 with
/// `allowGPUOptimizedContents` off, and 16 by 4, whose sample grid is 32 by 8. Metal makes
/// each in the same heap. No suite case and no boot is known to place one
#[test]
fn multisampled_textures_in_a_heap_over_guest_pages_die_where_placed() {
    use crate::os::OsVm;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::types::ResourceAddressRange;

    const SIZE: u64 = 0x10000;

    /// A `Shared` heap of `SIZE` bytes over pages of its own, and a 4-sample `RGBA8Unorm`
    /// render target of its options, 16 texels wide and `height` tall, its
    /// `allowGPUOptimizedContents` `allow`.
    fn heap_and_texture<P: Platform>(
        p: &P,
        d: &P::Device,
        (height, allow): (u64, Option<bool>),
    ) -> (P::Heap, P::TextureDescriptor) {
        // Left mapped: the heap's alias shares them until the process ends.
        let pages = OsVm::new().allocate_anywhere(SIZE).unwrap();
        let ranges = p
            .new_address_range_array(&[ResourceAddressRange {
                address: u64::try_from(pages.addr()).unwrap(),
                length: SIZE,
            }])
            .unwrap();
        let heap = d
            .new_heap(&HeapDesc {
                size: SIZE,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let multisample = p.texture_descriptor(&TextureDesc {
            pixel_format: 70,
            texture_type: 4,
            sample_count: 4,
            width: 16,
            height,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            allow_gpu_optimized_contents: allow,
            ..TextureDesc::default()
        });
        (heap, multisample)
    }

    let compressed = (16, None);
    let uncompressed = [(16, Some(false)), (4, None)];
    let (p, d) = (platform(), dev());
    let named = std::iter::once((compressed, "heap-texture-lossless-layout"))
        .chain(uncompressed.map(|shape| (shape, "HostHeapUncompressedMultisample")));
    for (shape, named) in named {
        let (heap, multisample) = heap_and_texture(&p, &d, shape);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| heap.new_texture(&multisample, 0).is_ok()).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newTextureWithDescriptor:offset:"),
            "{shape:?}: {line}"
        );
        assert!(line.contains(named), "{shape:?}: {line}");
    }
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        for shape in std::iter::once(compressed).chain(uncompressed) {
            let (heap, multisample) = heap_and_texture(&mp, &md, shape);
            assert!(
                heap.new_texture(&multisample, 0).unwrap().is_some(),
                "{shape:?}"
            );
        }
    }
}

/// A depth or stencil texture without allowGPUOptimizedContents placed in a heap, whose size
/// the runtime answers larger than Metal's (`optimized-depth-stencil-texture`,
/// [`crate::answer::HEARD`]). In a heap over no address ranges, core's legacy heap, one made
/// dies where it is made, at every shape, as every depth or stencil texture made there does
/// ([`HeapObject::legacy_made`]): a `Depth32Float` one and the suite's (a 2D
/// `Depth32Float_Stencil8` render target, 8x8, as `render_state_depth` and four more cases place
/// it there), and one with the property on, as core's copies would carry the runtime's layout to
/// the guest's pages. The suite's beside the first, where Metal's size (16512 bytes, as the
/// runtime's) runs past the heap's end, is nil, as Metal's; a placement inside the heap the
/// runtime answers nil dies where Metal's size fits (a 64x64 one at 0x4000 of the 0x8000 heap,
/// Metal's 16384 bytes, the runtime's 16512), and is nil where it does not either (a 128x128 one
/// at 0, Metal's 65536 bytes) and off Metal's alignment (a 16x16 one at 0x40), as the Metal
/// platform answers in a heap of its own; one past its end is nil. A buffer over all of the
/// heap, as core makes with each heap, is made first. In a `Shared` heap
/// over the guest's pages, as core's detach heap is, with no such buffer, a `Depth32Float` one
/// is made; a buffer and a texture over a byte of it die while it lives, and so does such a
/// texture over a byte of a buffer that lives. The Metal platform makes each made here or dying
/// here
#[test]
fn placements_of_optimized_depth_textures_die_where_the_size_decides() {
    use reims_vgpu_core::raise::death::DeathSite;
    let texture_sel = "newTextureWithDescriptor:offset:";
    /// A `Depth32Float` render target, `side` by `side`, of `options`.
    fn depth_in<P: Platform>(
        p: &P,
        side: u64,
        allow: Option<bool>,
        options: u64,
    ) -> P::TextureDescriptor {
        p.texture_descriptor(&TextureDesc {
            pixel_format: 252,
            width: side,
            height: side,
            usage: 0x4,
            resource_options: Some(options),
            allow_gpu_optimized_contents: allow,
            ..TextureDesc::default()
        })
    }
    fn depth_of<P: Platform>(p: &P, side: u64, allow: Option<bool>) -> P::TextureDescriptor {
        depth_in(p, side, allow, 0x120)
    }
    /// The suite's: a `Depth32Float_Stencil8` render target, 8x8, as its guests' records
    /// describe it, which serialize allowGPUOptimizedContents off.
    fn suites_of<P: Platform>(p: &P, options: u64) -> P::TextureDescriptor {
        p.texture_descriptor(&TextureDesc {
            pixel_format: 260,
            width: 8,
            height: 8,
            usage: 0x4,
            resource_options: Some(options),
            allow_gpu_optimized_contents: Some(false),
            ..TextureDesc::default()
        })
    }
    /// The offset past a texture of `size` bytes at offset 0, at its alignment `align` and a
    /// buffer's 0x100, where a buffer of 0x100 bytes is placed; and the first offset at its
    /// alignment where the texture covers that buffer's first byte.
    fn beside_and_over(size: u64, align: u64) -> (u64, u64) {
        let beside = size.next_multiple_of(align.max(0x100));
        (beside, (beside + 1 - size).next_multiple_of(align))
    }
    /// In `heap`, over the guest's pages, a `Depth32Float` one of `size` at 0, then a buffer
    /// over its last bytes, a texture over it, a buffer beside it and, once it is let go, another
    /// such texture over that buffer: `placed` takes each of those placements, by its case and
    /// selector, and whether it was made.
    type Placed<'a> = &'a dyn Fn(&str, &'static str, &mut dyn FnMut() -> bool);
    fn over_pages<P: Platform>(p: &P, heap: &P::Heap, size: SizeAndAlign, placed: Placed<'_>) {
        let buffer_sel = "newBufferWithLength:options:offset:";
        let texture_sel = "newTextureWithDescriptor:offset:";
        let options = heap.resource_options();
        let depth = depth_in(p, 16, Some(false), options);
        let (beside, over) = beside_and_over(size.size, size.align);
        let last = (size.size - 1) & !0xff;
        let t = heap.new_texture(&depth, 0).unwrap();
        assert!(t.is_some(), "the texture");
        placed("a buffer over its last bytes", buffer_sel, &mut || {
            (heap.new_buffer_of_length(0x100, options, last).unwrap()).is_some()
        });
        let linear = p.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 16,
            height: 16,
            usage: 0x13,
            resource_options: Some(options),
            ..TextureDesc::default()
        });
        placed("a texture over it", texture_sel, &mut || {
            (heap.new_texture(&linear, 0).unwrap()).is_some()
        });
        let b = heap.new_buffer_of_length(0x100, options, beside).unwrap();
        assert!(b.is_some(), "a buffer beside it");
        drop(t);
        placed("over a buffer that lives", texture_sel, &mut || {
            (heap.new_texture(&depth, over).unwrap()).is_some()
        });
        drop(b);
    }
    let depth = |side, allow| depth_of(&platform(), side, allow);
    let suites = suites_of(&platform(), 0x120);
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        let h = heap_of::<MetalPlatform>(&md, 0x8000, &options)
            .heap
            .unwrap();
        let made = h.new_texture(&depth_of(&mp, 16, Some(false)), 0).unwrap();
        assert!(made.is_some(), "Metal: a Depth32Float one at 0");
        drop(made);
        let made = h.new_texture(&suites_of(&mp, 0x120), 0).unwrap();
        assert!(made.is_some(), "Metal: the suite's at 0");
        drop(made);
        let answer = md
            .heap_texture_size_and_align(&suites_of(&mp, 0x120))
            .unwrap();
        let (beside, _) = beside_and_over(answer.size, answer.align);
        let past = h.new_texture(&suites_of(&mp, 0x120), beside).unwrap();
        assert!(past.is_none(), "Metal: the suite's at {beside:#x}");
        let fits = h
            .new_texture(&depth_of(&mp, 64, Some(false)), 0x4000)
            .unwrap();
        assert!(fits.is_some(), "Metal: a 64x64 one at 0x4000");
        drop(fits);
        let past = h.new_texture(&depth_of(&mp, 128, Some(false)), 0).unwrap();
        assert!(past.is_none(), "Metal: a 128x128 one at 0");
        let off = h
            .new_texture(&depth_of(&mp, 16, Some(false)), 0x40)
            .unwrap();
        assert!(off.is_none(), "Metal: a 16x16 one at 0x40");
        let (shared, ..) = heap_over_pages(&mp, &md);
        let desc = depth_in(&mp, 16, Some(false), shared.resource_options());
        let size = md.heap_texture_size_and_align(&desc).unwrap();
        over_pages(&mp, &shared, size, &|case, _, made| {
            assert!(made(), "Metal: {case}");
        });
    }
    let dies = |site: DeathSite, sel, case: &str| {
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{case}");
    };
    let d = dev();
    let raw = suites.fields().descriptor(texture_sel, Maker::SizeQuery);
    let answer = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
    let (beside, _) = beside_and_over(answer.size, answer.align);
    let h = private_heap();
    let whole = h.new_buffer_of_length(0x8000, 0x120, 0).unwrap().unwrap();
    for (case, desc, at) in [
        ("a Depth32Float one", depth(16, Some(false)), 0),
        ("the suite's", suites, 0),
        ("with the property on", depth(16, Some(true)), beside),
    ] {
        *crate::answer::LAST_LINE.lock() = None;
        dies(
            catch_death(|| h.new_texture(&desc, at)).unwrap_err(),
            texture_sel,
            case,
        );
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(
            line.contains("a heap over no address ranges"),
            "{case}: {line}"
        );
        assert!(line.contains("depth or stencil"), "{case}: {line}");
    }
    let suites = suites_of(&platform(), 0x120);
    assert!(
        (h.new_texture(&suites, beside).unwrap()).is_none(),
        "nil where Metal's size runs past the heap's end"
    );
    *crate::answer::LAST_LINE.lock() = None;
    dies(
        catch_death(|| h.new_texture(&depth(64, Some(false)), 0x4000)).unwrap_err(),
        texture_sel,
        "nil inside the heap where Metal's size fits",
    );
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert!(line.contains("which the runtime answers nil"), "{line}");
    assert!(
        (h.new_texture(&depth(128, Some(false)), 0).unwrap()).is_none(),
        "nil inside the heap where Metal's size does not fit either"
    );
    assert!(
        (h.new_texture(&depth(16, Some(false)), 0x40).unwrap()).is_none(),
        "nil off Metal's alignment"
    );
    assert!(
        h.new_texture(&depth(16, Some(false)), 0x8000)
            .unwrap()
            .is_none()
    );
    drop(whole);
    let p = platform();
    let (shared, ..) = heap_over_pages(&p, &d);
    let desc = depth_in(&p, 16, Some(false), shared.resource_options());
    let raw = desc.fields().descriptor(texture_sel, Maker::SizeQuery);
    let size = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
    let size = SizeAndAlign {
        size: size.size,
        align: size.align,
    };
    over_pages(&p, &shared, size, &|case, sel, made| {
        dies(catch_death(made).unwrap_err(), sel, case);
    });
}

/// An optimized `X32_Stencil8` render target (`optimized-depth-stencil-texture`) placed in a
/// heap over the guest's pages, whose size and alignment Metal's are not measured for
/// ([`crate::gpu::metal_space::metal_size_and_align`] names none), dies where the runtime makes
/// it, as the runtime's nil dies there: Metal may answer nil at an offset the runtime takes. A
/// `Depth32Float` one, measured, is made. Metal makes each
#[test]
fn unmeasured_optimized_placements_die_where_made() {
    let sel = "newTextureWithDescriptor:offset:";
    /// A render target of `pixel_format`, 16x16, of `heap`'s options, with
    /// allowGPUOptimizedContents off.
    fn target<P: Platform>(p: &P, heap: &P::Heap, pixel_format: u64) -> P::TextureDescriptor {
        p.texture_descriptor(&TextureDesc {
            pixel_format,
            width: 16,
            height: 16,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            allow_gpu_optimized_contents: Some(false),
            ..TextureDesc::default()
        })
    }
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    if let Some(md) = MetalPlatform::system_default_device() {
        for pixel_format in [252, 261] {
            let (heap, ..) = heap_over_pages(&mp, &md);
            let made = heap.new_texture(&target(&mp, &heap, pixel_format), 0);
            assert!(made.unwrap().is_some(), "Metal: {pixel_format}");
        }
    }
    let (p, d) = (platform(), dev());
    let (heap, ..) = heap_over_pages(&p, &d);
    let made = heap.new_texture(&target(&p, &heap, 252), 0).unwrap();
    assert!(made.is_some(), "Depth32Float");
    let (heap, ..) = heap_over_pages(&p, &d);
    let desc = target(&p, &heap, 261);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| heap.new_texture(&desc, 0)).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
    assert!(line.contains("not measured"), "{line}");
    assert!(line.contains("optimized-depth-stencil-texture"), "{line}");
}

/// A placement heap's buffer of length 0 is made at an offset of 256 bytes' alignment up to
/// the heap's end, of no bytes, at the offset into the heap; nil off the alignment and past
/// the end, as Metal's
#[test]
fn zero_length_heap_buffers_are_metals() {
    fn made<P: Platform>(d: &P::Device) -> Vec<Option<(u64, u64)>>
    where
        P::Device: GpuDevice<P>,
    {
        let h = heap_of::<P>(d, 0x10000, &[OptionSet::StorageMode(0)])
            .heap
            .unwrap();
        [0, 0x100, 0xff00, 0x10000, 0x10100, 100]
            .into_iter()
            .map(|offset| {
                let b = h.new_buffer_of_length(0, h.resource_options(), offset);
                b.unwrap()
                    .map(|b| (b.length(), b.gpu_address() - h.gpu_address()))
            })
            .collect()
    }
    let measured = [
        Some((0, 0)),
        Some((0, 0x100)),
        Some((0, 0xff00)),
        Some((0, 0x10000)),
        None,
        None,
    ];
    assert_eq!(made::<VulkanPlatform>(&dev()), measured);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(made::<MetalPlatform>(&md), measured);
    }
}

/// A 2D BGRA8Unorm descriptor of 16 x 16, Private, of usage `usage` (raw).
fn of_usage(usage: u64) -> TextureDesc {
    TextureDesc {
        pixel_format: 80,
        width: 16,
        height: 16,
        usage,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    }
}

/// The usage each creator's texture of `desc` answers on `d`: `newTextureWithDescriptor:`'s,
/// a view's of it, `newSharedTextureWithDescriptor:`'s, a texture's opened from that one's
/// handle, `newTextureWithDescriptor:iosurface:plane:`'s (plane 0 of `s`, whose storage
/// `desc_on` gives), a Shared buffer's two creators' (of `desc_on` too) and a Private
/// placement heap's; a swizzled view's of the first (blue and red swapped); and the heap's
/// size query.
fn usages<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    (desc, desc_on): (&P::TextureDescriptor, &P::TextureDescriptor),
    s: &OsSurface,
) -> ([u64; 8], u64, SizeAndAlign) {
    let buffer = d
        .new_buffer(BufferDesc {
            length: 1 << 16,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: None,
            no_copy: false,
            deallocator: None,
        })
        .unwrap()
        .unwrap();
    let heap = heap_of::<P>(d, 1 << 20, &[OptionSet::StorageMode(2)])
        .heap
        .unwrap();
    let plain = d.new_texture(OnFault::Return, desc).unwrap().unwrap();
    let shared = d.new_shared_texture(desc).unwrap().unwrap();
    let handle = shared.new_shared_texture_handle().unwrap().unwrap();
    let made = [
        plain.usage(),
        plain.new_texture_view(80).unwrap().unwrap().usage(),
        shared.usage(),
        d.open_shared_texture(Some(&handle), 0)
            .unwrap()
            .unwrap()
            .usage(),
        d.new_texture_on_surface(desc_on, Some(s), 0)
            .unwrap()
            .unwrap()
            .usage(),
        buffer
            .new_texture(OnFault::Return, desc_on, 0, 256)
            .unwrap()
            .unwrap()
            .usage(),
        buffer
            .new_linear_texture(desc_on, 0, 256, 4096)
            .unwrap()
            .unwrap()
            .usage(),
        heap.new_texture(desc, 0).unwrap().unwrap().usage(),
    ];
    // `MTLTextureSwizzleBlue`, `Green`, `Red`, `Alpha`.
    let swapped = Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    };
    let one = NsRange::new(0, 1);
    let swizzled = plain
        .new_swizzled_texture_subview(80, 2, one, one, swapped)
        .unwrap()
        .unwrap()
        .usage();
    (made, swizzled, d.heap_texture_size_and_align(desc).unwrap())
}

/// Every creator takes the usage bits Metal's public headers name none of (0x8, 0x4000,
/// 0x8000 and 0x10000, which a macOS 26.6.2 guest sets), making the texture as without them;
/// the texture, its views and a texture opened from its handle answer them but 0x8, a
/// swizzled view as Metal's does, and the heap's size query its size without them, as
/// Metal's (measured: [`crate::gpu::desc`])
#[test]
fn usage_bits_metal_takes_are_answered_as_metals() {
    let d = dev();
    let p = platform();
    let md = MetalPlatform::system_default_device();
    let s = bgra_surface(16, 16);
    let usage_of_all = |u: u64| [u & !0x8; 8];
    for usage in [0x1_0005, 0x1_0015, 0x4005, 0x8005, 0xd, 0x1_c01d] {
        let (desc, desc_on) = (of_usage(usage), of_usage(usage));
        let desc_on = TextureDesc {
            resource_options: Some(0),
            ..desc_on
        };
        let ours = usages::<VulkanPlatform>(
            &d,
            (
                &p.texture_descriptor(&desc),
                &p.texture_descriptor(&desc_on),
            ),
            &s,
        );
        assert_eq!(ours.0, usage_of_all(usage), "{usage:#x}");
        let without = d
            .heap_texture_size_and_align(&p.texture_descriptor(&of_usage(usage & 0x37)))
            .unwrap();
        assert_eq!(ours.2, without, "{usage:#x}");
        if let Some(md) = &md {
            let metal = (
                reims_vgpu_metal::gpu::desc::texture_descriptor(&desc),
                reims_vgpu_metal::gpu::desc::texture_descriptor(&desc_on),
            );
            let theirs = usages::<MetalPlatform>(md, (&metal.0, &metal.1), &s);
            assert_eq!(theirs, ours, "Metal: {usage:#x}");
        }
    }
}

/// The heap's size query answers a texture of `BlockWritesOnly` usage (0x10000) as Metal's
/// does, the size and alignment of one without the bit (1152 bytes for a 16 by 16
/// `BGRA8Unorm` texture of usage 0x10005 in `Private` storage, measured on the M3 Max), and
/// such a texture fits where a guest places it by that answer: a placement heap of the
/// answer's size makes it at offset 0. Of sizes, levels and array lengths here, in
/// `RGBA8Unorm`, `BGRA8Unorm` and `RGBA16Float`. A mipmapped one Metal keeps losslessly
/// compressed, of no `ShaderWrite` usage, answers Metal's as the runtime models it, with the
/// bit and without, and nothing is announced. Metal answers every texture here alike with the bit
/// and without
#[test]
fn block_writes_only_sizes_are_metals_and_fit() {
    let (d, p) = (dev(), platform());
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    crate::answer::RELAYED.lock().clear();
    for (format, (width, height), levels, slices) in [
        (80, (16, 16), 1, 1),
        (80, (100, 60), 7, 1),
        (115, (256, 256), 9, 1),
        (80, (64, 64), 1, 4),
        (70, (33, 17), 6, 3),
    ] {
        for usage in [0x1_0005, 0x1_0001, 0x1_0003] {
            let desc = TextureDesc {
                texture_type: if slices > 1 { 3 } else { 2 },
                pixel_format: format,
                width,
                height,
                mipmap_level_count: levels,
                array_length: slices,
                ..of_usage(usage)
            };
            let case = format!("{format} {width}x{height} {levels} {slices} {usage:#x}");
            let ours = d.heap_texture_size_and_align(&p.texture_descriptor(&desc));
            let without = TextureDesc {
                usage: usage & !0x1_0000,
                ..desc
            };
            let bare = d.heap_texture_size_and_align(&p.texture_descriptor(&without));
            let ours = ours.unwrap();
            assert_eq!(ours, bare.unwrap(), "{case}");
            if (format, width, levels, slices, usage) == (80, 16, 1, 1, 0x1_0005) {
                assert_eq!(ours.size, 1152, "{case}");
            }
            let heap = heap_of::<VulkanPlatform>(&d, ours.size, &options)
                .heap
                .unwrap();
            let made = heap.new_texture(&p.texture_descriptor(&desc), 0).unwrap();
            assert!(made.is_some(), "{case}");
            if let Some(md) = &md {
                let theirs = md.heap_texture_size_and_align(&mp.texture_descriptor(&desc));
                assert_eq!(theirs.unwrap(), ours, "Metal: {case}");
                let bare = md.heap_texture_size_and_align(&mp.texture_descriptor(&without));
                assert_eq!(bare.unwrap(), ours, "Metal without the bit: {case}");
            }
        }
    }
    let unmeasured = rt::Degradation::UnmeasuredHeapTextureSize;
    assert!(!crate::answer::RELAYED.lock().contains(&unmeasured));
}

/// A texture whose layout in Metal's heaps the runtime does not model, a PVRTC one, whose size
/// it answers as its device's image's (`unmeasured-heap-texture-size`), dies at each selector
/// that heard it, naming the kind and the texture, where the Metal platform answers:
/// `newTextureWithDescriptor:` of the suite's texture of each PVRTC format (`textures_formats`:
/// 2D, 16x16, one level, `Private`, `ShaderRead`, `allowGPUOptimizedContents` off) and of a
/// 32x32 one, the size query of the suite's texture, and its placement in a heap; a placement
/// at the heap's end is nil, as Metal's. A texture of 8 samples dies at the runtime's refusal
/// first (`SampleCount`), naming no kind, where Metal stops it (the runtime's measurement)
#[test]
fn unmodelled_heap_sizes_die() {
    let (sel, query, place) = (
        "newTextureWithDescriptor:",
        "heapTextureSizeAndAlignWithDescriptor:",
        "newTextureWithDescriptor:offset:",
    );
    let (p, d) = (platform(), dev());
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let suites = |pixel_format| TextureDesc {
        pixel_format,
        width: 16,
        height: 16,
        usage: 0x1,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let died = |at: &'static str, name: &str, f: &dyn Fn()| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(f).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, at),
            "{name}: {line}"
        );
        line
    };
    for pixel_format in 160..=167 {
        let t = suites(pixel_format);
        let name = format!("PVRTC {pixel_format}");
        if let Some(md) = &md {
            let theirs = md.new_texture(OnFault::Return, &mp.texture_descriptor(&t));
            assert!(theirs.unwrap().is_some(), "Metal: {name}");
        }
        let desc = p.texture_descriptor(&t);
        let line = died(sel, &name, &|| {
            drop(d.new_texture(OnFault::Return, &desc));
        });
        assert!(
            line.contains("unmeasured-heap-texture-size"),
            "{name}: {line}"
        );
        let format = format!("pixel format {pixel_format},");
        assert!(line.contains(&format), "{name}: {line}");
    }
    let t = suites(166);
    let wider = TextureDesc {
        width: 32,
        height: 32,
        ..t
    };
    if let Some(md) = &md {
        let metal = |t: &TextureDesc| mp.texture_descriptor(t);
        assert!(
            md.heap_texture_size_and_align(&metal(&t)).is_ok(),
            "Metal's query"
        );
        let heap = heap_of::<MetalPlatform>(md, 0x8000, &options).heap.unwrap();
        assert!(
            heap.new_texture(&metal(&t), 0).unwrap().is_some(),
            "Metal's placement"
        );
        assert!(
            heap.new_texture(&metal(&t), 0x8000).unwrap().is_none(),
            "Metal's end"
        );
        let made = md.new_texture(OnFault::Return, &metal(&wider));
        assert!(made.unwrap().is_some(), "Metal's 32x32");
    }
    let desc = p.texture_descriptor(&t);
    let heap = heap_of::<VulkanPlatform>(&d, 0x8000, &options)
        .heap
        .unwrap();
    assert!(
        heap.new_texture(&desc, 0x8000).unwrap().is_none(),
        "the end"
    );
    let wide = p.texture_descriptor(&wider);
    for (at, name, line) in [
        (
            query,
            "the query",
            died(query, "the query", &|| {
                drop(d.heap_texture_size_and_align(&desc))
            }),
        ),
        (
            place,
            "the placement",
            died(place, "the placement", &|| drop(heap.new_texture(&desc, 0))),
        ),
        (
            sel,
            "32x32",
            died(sel, "32x32", &|| {
                drop(d.new_texture(OnFault::Return, &wide))
            }),
        ),
    ] {
        assert!(
            line.contains("unmeasured-heap-texture-size"),
            "{at} {name}: {line}"
        );
        assert!(line.contains("pixel format 166"), "{at} {name}: {line}");
    }
    let eight = p.texture_descriptor(&TextureDesc {
        texture_type: 4,
        pixel_format: 70,
        width: 16,
        height: 16,
        sample_count: 8,
        usage: 0x4,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let made = || drop(d.new_texture(OnFault::Return, &eight));
    let sized = || drop(d.heap_texture_size_and_align(&eight));
    for (at, f) in [(sel, &made as &dyn Fn()), (query, &sized)] {
        let line = died(at, "8 samples", f);
        assert!(line.contains("SampleCount"), "8 samples at {at}: {line}");
        assert!(!line.contains("unmeasured-heap-texture-size"), "{line}");
    }
}

/// Asks Metal's `heapTextureSizeAndAlignWithDescriptor:` of PVRTC textures (each of the eight
/// formats) of 2D, 2D array, cube and cube array types, of each power-of-two side from 4 to
/// 2048 (Metal asserts at a PVRTC descriptor of other sides), of one level and of the full chain, at `Private` and `Shared` storage, of usage `ShaderRead`,
/// with allowGPUOptimizedContents off and on, against the runtime's answer, its device's
/// image's (`unmeasured-heap-texture-size`), which `newTextureWithDescriptor:` would size the
/// texture's range of Metal's space by. Prints, for each format and storage, how many rows the
/// runtime answers as Metal, and up to 12 it answers otherwise. A measurement for the
/// runtime's model of PVRTC heap sizes, which it does not have: on MoltenVK the device's image
/// is a Metal texture, so its answer may be Metal's; on another device it is not.
#[test]
#[ignore = "a measurement: Metal's PVRTC heap sizes against the runtime's"]
fn pvrtc_sizes_against_the_runtimes() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let (p, d) = (platform(), dev());
    let mut rows = Vec::new();
    for side in (2..=11).map(|power| 1u64 << power) {
        for (texture_type, slices) in [(2, 1), (3, 2), (3, 3), (5, 1), (6, 2)] {
            let (width, height) = (side, side);
            for levels in [1, u64::from(side.ilog2()) + 1] {
                for allow in [false, true] {
                    rows.push(TextureDesc {
                        texture_type,
                        width,
                        height,
                        mipmap_level_count: levels,
                        array_length: slices,
                        usage: 0x1,
                        allow_gpu_optimized_contents: Some(allow),
                        ..TextureDesc::default()
                    });
                }
            }
        }
    }
    let mut unequal = 0;
    for pixel_format in 160..=167 {
        for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
            let (mut other, mut equal) = (Vec::new(), 0);
            for row in &rows {
                let mut t = *row;
                t.pixel_format = pixel_format;
                t.resource_options = Some(options);
                let theirs = metal
                    .heap_texture_size_and_align(&mp.texture_descriptor(&t))
                    .unwrap();
                let raw = (p.texture_descriptor(&t).fields()).descriptor(sel, Maker::SizeQuery);
                let ours =
                    plain(|| d.raw.heap_texture_size_and_align(&raw)).map(|answer| SizeAndAlign {
                        size: answer.size,
                        align: answer.align,
                    });
                match ours {
                    Ok(ours) if ours == theirs => equal += 1,
                    ours => other.push(format!(
                        "  type {} {}x{} levels {} slices {} allow {:?}: Metal's {theirs:x?}, \
                         the runtime's {ours:x?}",
                        t.texture_type,
                        t.width,
                        t.height,
                        t.mipmap_level_count,
                        t.array_length,
                        t.allow_gpu_optimized_contents,
                    )),
                }
            }
            eprintln!(
                "PVRTC {pixel_format} {storage}: the runtime's equal at {equal} of {} rows, \
                 other at {}",
                rows.len(),
                other.len()
            );
            for row in other.iter().take(12) {
                eprintln!("{row}");
            }
            unequal += other.len();
        }
    }
    eprintln!("unequal: {unequal}");
}

/// A descriptor takes any usage bit at `setUsage:`, as Metal's does, and drops it at the next;
/// every creator, and the heap's size query, of one Metal does not define dies there, as Metal
/// asserts (measured: [`crate::gpu::desc`]; Metal's side cannot run, as its assert ends the
/// process), but `newSharedTextureWithDescriptor:` of a format Metal makes no texture of,
/// which answers nil first, as Metal's does; once `setUsage:` drops the bit, every creator
/// makes the texture
#[test]
fn usage_bits_metal_does_not_define_die_where_textures_are_made() {
    let d = dev();
    let p = platform();
    let md = MetalPlatform::system_default_device();
    let s = bgra_surface(16, 16);
    let heap = private_heap();
    let buffer = d
        .new_buffer(BufferDesc {
            length: 1 << 16,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: None,
            no_copy: false,
            deallocator: None,
        })
        .unwrap()
        .unwrap();
    for usage in [0x45, 0x2_0005, 0x8000_0000_0000_0001] {
        let desc = p.texture_descriptor(&of_usage(5));
        let desc_on = p.texture_descriptor(&TextureDesc {
            resource_options: Some(0),
            ..of_usage(5)
        });
        desc.set_usage(usage);
        desc_on.set_usage(usage);
        let deaths = [
            (
                "newTextureWithDescriptor:",
                catch_death(|| d.new_texture(OnFault::Return, &desc)).map(drop),
            ),
            (
                "newSharedTextureWithDescriptor:",
                catch_death(|| d.new_shared_texture(&desc)).map(drop),
            ),
            (
                "heapTextureSizeAndAlignWithDescriptor:",
                catch_death(|| d.heap_texture_size_and_align(&desc)).map(drop),
            ),
            (
                "newTextureWithDescriptor:iosurface:plane:",
                catch_death(|| d.new_texture_on_surface(&desc_on, Some(&s), 0)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:bytesPerRow:",
                catch_death(|| buffer.new_texture(OnFault::Return, &desc_on, 0, 256)).map(drop),
            ),
            (
                "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                catch_death(|| buffer.new_linear_texture(&desc_on, 0, 256, 4096)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:",
                catch_death(|| heap.new_texture(&desc, 0)).map(drop),
            ),
        ];
        for (sel, death) in deaths {
            let site = death.unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "{usage:#x}"
            );
        }
        for format in [263, 556, 651, 652] {
            let none = TextureDesc {
                pixel_format: format,
                ..of_usage(5)
            };
            let ours = p.texture_descriptor(&none);
            ours.set_usage(usage);
            let shared = d.new_shared_texture(&ours).unwrap();
            assert!(shared.is_none(), "{format} {usage:#x}");
            if let Some(md) = &md {
                let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&none);
                metal.set_usage(usage);
                let shared = md.new_shared_texture(&metal).unwrap();
                assert!(shared.is_none(), "Metal: {format} {usage:#x}");
            }
        }
        desc.set_usage(5);
        desc_on.set_usage(5);
        let made = [
            d.new_texture(OnFault::Return, &desc).unwrap().is_some(),
            d.new_shared_texture(&desc).unwrap().is_some(),
            d.heap_texture_size_and_align(&desc).unwrap().size > 0,
            d.new_texture_on_surface(&desc_on, Some(&s), 0)
                .unwrap()
                .is_some(),
            buffer
                .new_texture(OnFault::Return, &desc_on, 0, 256)
                .unwrap()
                .is_some(),
            buffer
                .new_linear_texture(&desc_on, 0, 256, 4096)
                .unwrap()
                .is_some(),
            heap.new_texture(&desc, 0).unwrap().is_some(),
        ];
        assert_eq!(made, [true; 7], "{usage:#x}");
    }
}

/// The `uniqueIdentifier` of each texture of a sequence of creations on platform `P`, with
/// what each step made: textures of the device's creator, a view, a heap's, a buffer's, over a
/// surface and shareable, beside a buffer, a heap, a sampler state, a command queue, a nil
/// texture and a release, then, after another release, one on another thread and one on this
/// thread again.
fn texture_identifiers_on<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<(&'static str, u64, u64)>
where
    P::Texture: Send,
    P::Device: Sync,
{
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
    let of = |format, options| TextureDesc {
        pixel_format: format,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(options),
        ..TextureDesc::default()
    };
    let private = desc(&of(80, 0x20));
    let shared = desc(&of(80, 0));
    let texture = || d.new_texture(OnFault::Return, &private).unwrap().unwrap();
    let mut ids = Vec::new();
    let t1 = texture();
    ids.push(("t1", t1.unique_identifier(), t1.gpu_resource_id()));
    let t2 = texture();
    ids.push(("t2", t2.unique_identifier(), t2.gpu_resource_id()));
    let buffer = d
        .new_buffer_of_length(OnFault::Return, 4096, 0)
        .unwrap()
        .unwrap();
    let t3 = texture();
    ids.push((
        "t3, after a buffer",
        t3.unique_identifier(),
        t3.gpu_resource_id(),
    ));
    let view = t1.new_texture_view(81).unwrap().unwrap();
    ids.push((
        "a view of t1",
        view.unique_identifier(),
        view.gpu_resource_id(),
    ));
    let t4 = texture();
    ids.push((
        "t4, after the view",
        t4.unique_identifier(),
        t4.gpu_resource_id(),
    ));
    let heap = d
        .new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let t5 = texture();
    ids.push((
        "t5, after a heap",
        t5.unique_identifier(),
        t5.gpu_resource_id(),
    ));
    let placed = heap.new_texture(&private, 0).unwrap().unwrap();
    ids.push((
        "the heap's texture",
        placed.unique_identifier(),
        placed.gpu_resource_id(),
    ));
    let over = buffer
        .new_texture(OnFault::Return, &shared, 0, 256)
        .unwrap()
        .unwrap();
    ids.push((
        "a texture over the buffer",
        over.unique_identifier(),
        over.gpu_resource_id(),
    ));
    let surface = bgra_surface(4, 4);
    let on = d
        .new_texture_on_surface(&shared, Some(&surface), 0)
        .unwrap()
        .unwrap();
    ids.push((
        "a texture over a surface",
        on.unique_identifier(),
        on.gpu_resource_id(),
    ));
    let sharable = d.new_shared_texture(&private).unwrap().unwrap();
    ids.push((
        "a shareable texture",
        sharable.unique_identifier(),
        sharable.gpu_resource_id(),
    ));
    let _sampler = d.new_sampler_state(&SamplerDesc::default()).unwrap();
    let t6 = texture();
    ids.push((
        "t6, after a sampler state",
        t6.unique_identifier(),
        t6.gpu_resource_id(),
    ));
    drop(t2);
    let t7 = texture();
    ids.push((
        "t7, after t2's release",
        t7.unique_identifier(),
        t7.gpu_resource_id(),
    ));
    assert!(
        d.new_texture(OnFault::Return, &desc(&of(263, 0x20)))
            .unwrap()
            .is_none()
    );
    let t8 = texture();
    ids.push((
        "t8, after a nil texture",
        t8.unique_identifier(),
        t8.gpu_resource_id(),
    ));
    let _queue = d.new_command_queue().unwrap();
    let t9 = texture();
    ids.push((
        "t9, after a command queue",
        t9.unique_identifier(),
        t9.gpu_resource_id(),
    ));
    // A release first, so the lowest free ID is not the next one.
    drop(t8);
    let t10 = std::thread::scope(|s| s.spawn(texture).join().unwrap());
    ids.push((
        "t10, on another thread, after t8's release",
        t10.unique_identifier(),
        t10.gpu_resource_id(),
    ));
    let t11 = texture();
    ids.push((
        "t11, on this thread again",
        t11.unique_identifier(),
        t11.gpu_resource_id(),
    ));
    ids.push(("t1 again", t1.unique_identifier(), t1.gpu_resource_id()));
    ids.push((
        "the view again",
        view.unique_identifier(),
        view.gpu_resource_id(),
    ));
    ids
}

/// Each texture's `uniqueIdentifier` and `gpuResourceID` on `d`, over releases in several
/// orders, a release while a command buffer holds the texture, a view's release, and textures
/// placed at an index and forced to one.
fn texture_reuse_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<(String, u64, u64)> {
    use reims_vgpu_core::platform::blit::BlitCmd;
    use reims_vgpu_core::platform::blit::BlitEncoder as _;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::types::{ClearColor, MtlRegion};
    let private = desc(&TextureDesc {
        pixel_format: 80,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let texture = || d.new_texture(OnFault::Return, &private).unwrap().unwrap();
    let mut ids = Vec::new();
    let mut note = |what: &str, t: &P::Texture| {
        ids.push((what.to_string(), t.unique_identifier(), t.gpu_resource_id()));
    };
    let mut ts: Vec<Option<P::Texture>> = (0..6).map(|_| Some(texture())).collect();
    for (i, t) in ts.iter().enumerate() {
        note(&format!("t{i}"), t.as_ref().unwrap());
    }
    ts[1] = None;
    ts[3] = None;
    let a = texture();
    note("after releasing t1 then t3", &a);
    let b = texture();
    note("and another", &b);
    ts[4] = None;
    ts[2] = None;
    let c = texture();
    note("after releasing t4 then t2", &c);
    let e = texture();
    note("and another", &e);
    let f = texture();
    note("and a third", &f);
    ts[5] = None;
    let g = texture();
    note("after releasing t5, the highest", &g);
    let q = d.new_command_queue().unwrap();
    let held = texture();
    note("held by a command buffer", &held);
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let en = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    for _ in 0..2000 {
        en.encode(
            OnFault::Return,
            BlitCmd::FillTextureColor {
                texture: &held,
                level: 0,
                slice: 0,
                region: MtlRegion {
                    origin: reims_vgpu_core::platform::types::MtlOrigin { x: 0, y: 0, z: 0 },
                    size: reims_vgpu_core::platform::types::MtlSize {
                        width: 4,
                        height: 4,
                        depth: 1,
                    },
                },
                color: ClearColor::default(),
                pixel_format: 80,
            },
        )
        .unwrap();
    }
    en.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    drop(held);
    let h = texture();
    note("after releasing it in flight", &h);
    cb.wait_until_completed(OnFault::Return).unwrap();
    let i = texture();
    note("after the command buffer completed", &i);
    drop(cb);
    let j = texture();
    note("after the command buffer's release", &j);
    let view = a.new_texture_view(81).unwrap().unwrap();
    note("a view", &view);
    drop(view);
    let k = texture();
    note("after the view's release", &k);
    let placed = |index| {
        let desc = desc(&TextureDesc {
            pixel_format: 80,
            width: 4,
            height: 4,
            usage: 1,
            resource_options: Some(0x20),
            resource_index: Some(index),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Return, &desc).unwrap().unwrap()
    };
    let p1 = placed((100, false));
    note("placed at 100", &p1);
    let p2 = placed((200, true));
    note("placed at 200, forced", &p2);
    let l = texture();
    note("after the placed ones", &l);
    drop(p1);
    let m = texture();
    note("after releasing the one at 100", &m);
    ids
}

/// A texture's `uniqueIdentifier` is its `gpuResourceID`, and both are Metal's, from a
/// process's start, through every creator's texture (the device's, a view, a heap's, one over
/// a buffer and one over a surface, a shareable one), beside a buffer, a heap, a sampler
/// state, a command queue and a nil texture, after a release, and on another thread after a
/// release, where the lowest free ID is not the next one, and on this thread again. Metal's
/// side runs first, in a process where nothing else has used Metal's device yet
/// (`texture_identifiers_are_reused_as_metals` says why)
#[test]
fn texture_identifiers_are_metals() {
    let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let theirs = MetalPlatform::system_default_device()
        .map(|md| texture_identifiers_on::<MetalPlatform>(&md, &metal_desc));
    let p = crate::platform::tests::platform();
    let ours = texture_identifiers_on::<VulkanPlatform>(&dev(), &|t| p.texture_descriptor(t));
    for (what, unique, id) in &ours {
        assert_eq!(unique, id, "{what}");
    }
    if let Some(theirs) = theirs {
        assert_eq!(ours, theirs);
    }
}

/// Texture identifiers are reused as Metal reuses them, from a process's start: the lowest
/// free first, over releases in several orders; one released while a command buffer holds the
/// texture once it completed; a view's at its release; and a texture placed at an index or
/// forced to one takes it, the next plain one going on below. Metal's side runs first, in a
/// process where nothing else has used Metal's device yet, as in a process that runs the
/// Metal platform alone: there a texture a command buffer held is free when
/// `waitUntilCompleted` returns (measured on the M3 Max: 5 runs of 5, with and without an
/// autorelease pool over the encoding). Once this platform's run of the sequence has put
/// MoltenVK's work on Metal's device, Metal frees it later, a race of its own: still held
/// after the wait in 3 runs of 3 (after the command buffer's release too in two of them), and
/// free 300 ms after the wait in 2 runs of 2
#[test]
fn texture_identifiers_are_reused_as_metals() {
    let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let theirs = MetalPlatform::system_default_device()
        .map(|md| texture_reuse_on::<MetalPlatform>(&md, &metal_desc));
    let p = crate::platform::tests::platform();
    let ours = texture_reuse_on::<VulkanPlatform>(&dev(), &|t| p.texture_descriptor(t));
    for (what, unique, id) in &ours {
        assert_eq!(unique, id, "{what}");
    }
    if let Some(theirs) = theirs {
        assert_eq!(ours, theirs);
    }
}

/// Each texture's `uniqueIdentifier` and `gpuResourceID` on `d` where Metal hands out an ID a
/// live texture holds a second time: after the device's first command
/// queue and a reservation of textures 5 and 40, sixteen textures (IDs 1 to 4 twice first,
/// measured here: `gpu::device`'s `a_texture_takes_a_live_ones_id_where_metal_gives_it`), then,
/// each followed by new textures, the release of one of an ID's two textures, of the other,
/// of both of another ID's in turn, and of two IDs' textures alternately.
fn texture_reissue_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<(String, u64, u64)> {
    use reims_vgpu_core::platform::resource::Texture as _;
    let private = desc(&TextureDesc {
        pixel_format: 80,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let _queue = d.new_command_queue().unwrap();
    d.reserve_resource_indices(0, &[5, 40]).unwrap();
    let texture = || d.new_texture(OnFault::Return, &private).unwrap().unwrap();
    let mut ids = Vec::new();
    let mut note = |what: String, t: &P::Texture| {
        ids.push((what, t.unique_identifier(), t.gpu_resource_id()));
    };
    let mut ts: Vec<Option<P::Texture>> = (0..16).map(|_| Some(texture())).collect();
    for (i, t) in ts.iter().enumerate() {
        note(format!("t{i}"), t.as_ref().unwrap());
    }
    let mut later = Vec::new();
    for (step, released, made) in [
        ("t1, one of ID 2's two", &[1][..], 1),
        ("t5, the other", &[5][..], 2),
        ("t2 then t6, both of ID 3's", &[2, 6][..], 2),
        (
            "t0, t3, t4 and t7, IDs 1 and 4 alternately",
            &[0, 3, 4, 7][..],
            4,
        ),
    ] {
        for &i in released {
            ts[i] = None;
        }
        for n in 0..made {
            let t = texture();
            note(format!("after releasing {step}: {n}"), &t);
            later.push(t);
        }
    }
    for n in 0..4 {
        let t = texture();
        note(format!("then {n}"), &t);
        later.push(t);
    }
    std::mem::forget((ts, later));
    ids
}

/// A texture takes an ID a live texture holds where Metal hands it out a second time, and
/// IDs come back as Metal's pool gives them back, over releases of either or both of an ID's
/// textures in several orders, from a process's start; each texture's
/// `uniqueIdentifier` is its ID throughout, as Metal's. Metal's side runs first, before
/// MoltenVK makes objects of its own on Metal's device
#[test]
fn texture_identifiers_are_reissued_as_metals() {
    let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let theirs = MetalPlatform::system_default_device()
        .map(|md| texture_reissue_on::<MetalPlatform>(&md, &metal_desc));
    let p = crate::platform::tests::platform();
    let ours = texture_reissue_on::<VulkanPlatform>(&dev(), &|t| p.texture_descriptor(t));
    for (what, unique, id) in &ours {
        assert_eq!(unique, id, "{what}");
    }
    if let Some(theirs) = theirs {
        assert_eq!(ours, theirs);
    }
}

/// A format's block: its bytes, and its width and height in texels.
type FormatBlock = (u64, (u64, u64));

/// A texture's shape, placed in a heap over no address ranges by [`legacy_heap_bytes_on`].
#[derive(Clone, Copy, Debug)]
struct LegacyShape {
    format: u64,
    /// A block's bytes, a texel's for a format not of blocks.
    block_bytes: u64,
    /// A block's width and height in texels.
    block: (u64, u64),
    texture_type: u64,
    width: u64,
    height: u64,
    depth: u64,
    levels: u64,
    array: u64,
    usage: u64,
    optimized: Option<bool>,
}

impl LegacyShape {
    /// A 2D texture of one level of `format` (a block of `block_bytes` bytes and `block`
    /// texels), `width` by `height` texels.
    fn flat(
        format: u64,
        (block_bytes, block): FormatBlock,
        (width, height): (u64, u64),
        usage: u64,
        optimized: Option<bool>,
    ) -> LegacyShape {
        LegacyShape {
            format,
            block_bytes,
            block,
            texture_type: 2,
            width,
            height,
            depth: 1,
            levels: 1,
            array: 1,
            usage,
            optimized,
        }
    }
}

/// What `P`'s device `d` shows of a texture of `shape` at `offset` of a `Private`, untracked
/// placement heap over no address ranges, made as core makes its legacy heap, with a `Private`
/// buffer over all of the heap made first, as core's: the texture's size; the heap's bytes over
/// the texture's span after every level and slice is copied from a `Shared` buffer of one
/// pattern, read through the heap's buffer, as core copies them to the guest's pages; then each
/// level's and slice's texels, copied to a buffer, after another pattern is copied into the
/// heap's buffer over the span, as core copies the guest's pages in. Every copy is a blit of its
/// own command buffer. `None` where the device answers nil for the texture.
fn legacy_heap_bytes_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    shape: LegacyShape,
    offset: u64,
) -> Option<(u64, Vec<u8>, Vec<u8>)>
where
    P::Device: GpuDevice<P>,
{
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _};

    const HEAP: u64 = 0x40_0000;
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let heap = heap_of::<P>(d, HEAP, &options).heap.unwrap();
    let cover = heap.new_buffer_of_length(HEAP, 0x20, 0).unwrap().unwrap();
    let texture_desc = desc(&TextureDesc {
        texture_type: shape.texture_type,
        pixel_format: shape.format,
        width: shape.width,
        height: shape.height,
        depth: shape.depth,
        mipmap_level_count: shape.levels,
        array_length: shape.array,
        usage: shape.usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: shape.optimized,
        ..TextureDesc::default()
    });
    let size = d.heap_texture_size_and_align(&texture_desc).unwrap().size;
    let t = heap.new_texture(&texture_desc, offset).unwrap()?;
    let shared = |length: u64, byte: &dyn Fn(usize) -> u8| {
        let b = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer of `length` bytes, which no command uses yet.
        let s = unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr::<u8>(), n) };
        for (i, v) in s.iter_mut().enumerate() {
            *v = byte(i);
        }
        b
    };
    let bytes = |b: &P::Buffer, length: u64| {
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer's first `length` bytes, read after its copies completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    };
    let q = d.new_command_queue().unwrap();
    let run = |cmds: &mut dyn FnMut(&P::BlitEncoder)| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        cmds(&e);
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    // Each level's and slice's region, its rows' and images' bytes and its place, packed.
    let slices = if shape.texture_type == 5 { 6 } else { 1 } * shape.array;
    let mut parts = Vec::new();
    let mut packed = 0;
    for level in 0..shape.levels {
        let (w, h, z) = (
            (shape.width >> level).max(1),
            (shape.height >> level).max(1),
            (shape.depth >> level).max(1),
        );
        let (bw, bh) = shape.block;
        let row = w.div_ceil(bw) * shape.block_bytes;
        let image = row * h.div_ceil(bh);
        for slice in 0..slices {
            parts.push((level, slice, MtlSize::new(w, h, z), row, image, packed));
            packed += image * z;
        }
    }
    let source = shared(packed, &|i| (i * 13 + 5) as u8);
    run(&mut |e| {
        for &(level, slice, size, row, image, at) in &parts {
            let copy = BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: at,
                source_bytes_per_row: row,
                source_bytes_per_image: image,
                source_size: size,
                destination_texture: &t,
                destination_slice: slice,
                destination_level: level,
                destination_origin: MtlOrigin::new(0, 0, 0),
            };
            e.encode(OnFault::Die, copy).unwrap();
        }
    });
    let span = |from: &P::Buffer, from_at: u64, to: &P::Buffer, to_at: u64| {
        run(&mut |e| {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset: from_at,
                destination_buffer: Some(to),
                destination_offset: to_at,
                size,
            };
            e.encode(OnFault::Die, copy).unwrap();
        });
    };
    let back = shared(size, &|_| 0);
    span(&cover, offset, &back, 0);
    let seen = bytes(&back, size);
    span(&shared(size, &|i| (i * 7 + 3) as u8), 0, &cover, offset);
    let texels = shared(packed, &|_| 0);
    run(&mut |e| {
        for &(level, slice, size, row, image, at) in &parts {
            let copy = BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: slice,
                source_level: level,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: size,
                destination_buffer: &texels,
                destination_offset: at,
                destination_bytes_per_row: row,
                destination_bytes_per_image: image,
            };
            e.encode(OnFault::Die, copy).unwrap();
        }
    });
    Some((size, seen, bytes(&texels, packed)))
}

/// Every colour format Metal describes on this host that the runtime makes textures of, with
/// its block's bytes and extent: neither a depth or stencil format, nor a two-plane one (550,
/// 551, which [`Placements`] keeps apart), nor a PVRTC one, whose size the runtime does not model
/// ([`crate::gpu::metal_space::unmeasured_size`]), nor one of 630 to 645, which an Apple M4 under
/// macOS 27.0.1 describes and neither of the runtime's levels does, whose layout this platform's
/// rules take from macOS 26.7 ([`crate::gpu::format_info`]).
fn legacy_heap_formats() -> Vec<(u64, FormatBlock)> {
    let sel = "legacy_heap_formats";
    (1..2048u64)
        .filter(|&raw| crate::gpu::format_info::described(sel, raw))
        .filter(|&raw| {
            let pf = u32::try_from(raw)
                .ok()
                .and_then(|raw| PixelFormat::try_from(raw).ok());
            pf.is_some_and(|pf| !metal2vulkan::metal::format::is_depth_stencil(pf))
        })
        .filter(|&raw| !matches!(raw, 160..=167 | 550 | 551))
        .filter(|&raw| !crate::gpu::format_info::level_decides(raw))
        .filter_map(|raw| {
            let block = crate::gpu::format_info::block(sel, raw, "a heap texture of");
            let [bw, bh] = block.extent();
            (block.bytes() > 0).then_some((raw, (block.bytes(), (bw, bh))))
        })
        .collect()
}

/// What [`legacy_heap_bytes_on`] answers.
type LegacyBytes = Option<(u64, Vec<u8>, Vec<u8>)>;

/// What [`legacy_heap_bytes_on`] shows of `shape` at 0x10000 on the Vulkan platform's `d`: `None`
/// where it dies of another kind than the heap's (the format, which the runtime lacks), the
/// death's line where it dies of the heap's.
fn legacy_heap_death(d: &VulkanDevice, shape: LegacyShape) -> Option<Result<LegacyBytes, String>> {
    let desc = |t: &TextureDesc| platform().texture_descriptor(t);
    *crate::answer::LAST_LINE.lock() = None;
    match catch_death(|| legacy_heap_bytes_on::<VulkanPlatform>(d, &desc, shape, 0x10000)) {
        Ok(bytes) => Some(Ok(bytes)),
        Err(site) => {
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            let heaps = "placed in a heap over no address ranges";
            let at = (DeathKind::Abort, "newTextureWithDescriptor:offset:");
            ((site.kind, site.file) == at && line.contains(heaps)).then_some(Err(line))
        }
    }
}

/// A texture of one sample and a colour format that Metal keeps uncompressed
/// ([`metal_keeps_plain`]), placed in a heap over no address ranges with a buffer over all of
/// the heap first, as core makes its legacy heap, is made on MoltenVK, and the heap's bytes over
/// it are Metal's both ways ([`legacy_heap_bytes_on`]): the bytes copied out after its texels
/// are written, and the texels read after other bytes are copied in, byte for byte. Of every
/// colour format the runtime makes on this host: a 2D texture without
/// `allowGPUOptimizedContents` of 64 by 64 blocks, and of `RenderTarget` usage too; one of 8 by
/// 8 texels, or one block where that is larger, with the property; one of `PixelFormatView`
/// usage with it; one of unknown usage with it. Of `RGBA8Unorm`, `R32Uint`, `RGBA16Float` and
/// `BC1_RGBA` without the property: a mipmapped 2D texture, a 2D array, a cube, a cube array, a
/// 1D texture and a 1D array, and a 3D texture of the plain formats; with the property, render
/// targets whose shorter side is 15 texels, a 1D one and an `R32Uint` of `ShaderAtomic` usage.
/// A texture a death of the format refuses (one the runtime lacks) is left out: 589 are compared
/// on an Apple M3 Max under macOS 26.7
#[test]
fn legacy_heap_textures_are_metals_where_metal_keeps_them_plain() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let d = dev();
    if d.raw.properties().description.driver_id != Some(metal2vulkan::vk::DriverId::MOLTENVK) {
        return;
    }
    let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let mut shapes = Vec::new();
    for (format, block) in legacy_heap_formats() {
        let (_, (bw, bh)) = block;
        let big = (bw * 64, bh * 64);
        let small = (bw * (8 / bw).max(1), bh * (8 / bh).max(1));
        let compressed = (bw, bh) != (1, 1);
        shapes.push(LegacyShape::flat(format, block, big, 0x1, Some(false)));
        shapes.push(LegacyShape::flat(format, block, small, 0x1, None));
        shapes.push(LegacyShape::flat(format, block, big, 0x11, None));
        if !compressed {
            shapes.push(LegacyShape::flat(format, block, big, 0x5, Some(false)));
            shapes.push(LegacyShape::flat(format, block, big, 0x0, None));
        }
    }
    let kinds = [
        (2, 128, 64, 1, 8, 1),
        (3, 32, 32, 1, 1, 3),
        (5, 32, 32, 1, 6, 1),
        (6, 32, 32, 1, 1, 2),
        (0, 256, 1, 1, 1, 1),
        (1, 256, 1, 1, 1, 4),
        (7, 16, 16, 8, 1, 1),
    ];
    for (format, block) in [
        (70, (4, (1, 1))),
        (53, (4, (1, 1))),
        (115, (8, (1, 1))),
        (130, (8, (4, 4))),
    ] {
        for (texture_type, width, height, depth, levels, array) in kinds {
            if block.1 != (1, 1) && matches!(texture_type, 0 | 1 | 7) {
                continue;
            }
            shapes.push(LegacyShape {
                texture_type,
                width,
                height,
                depth,
                levels,
                array,
                ..LegacyShape::flat(format, block, (0, 0), 0x1, Some(false))
            });
        }
        if block.1 == (1, 1) {
            shapes.push(LegacyShape::flat(format, block, (15, 64), 0x5, None));
            shapes.push(LegacyShape::flat(format, block, (64, 15), 0x5, None));
            shapes.push(LegacyShape {
                texture_type: 0,
                ..LegacyShape::flat(format, block, (256, 1), 0x5, None)
            });
        }
    }
    shapes.push(LegacyShape::flat(53, (4, (1, 1)), (64, 64), 0x25, None));
    let mut compared = 0;
    for shape in shapes {
        let case = format!("{shape:?}");
        let Some(made) = legacy_heap_death(&d, shape) else {
            continue;
        };
        let ours = made.unwrap_or_else(|line| panic!("{case}: {line}"));
        let theirs = legacy_heap_bytes_on::<MetalPlatform>(&metal, &metal_desc, shape, 0x10000);
        assert_eq!(ours.is_some(), theirs.is_some(), "{case}");
        let (Some(ours), Some(theirs)) = (ours, theirs) else {
            continue;
        };
        assert_eq!(ours.0, theirs.0, "size: {case}");
        assert!(ours.1 == theirs.1, "copied out: {case}");
        assert!(ours.2 == theirs.2, "copied in: {case}");
        compared += 1;
    }
    assert!(compared >= 500, "{compared} compared");
}

/// A colour texture placed in a heap over no address ranges whose bytes the rule of
/// [`HeapObject::legacy_made`] does not show are Metal's is made where Metal makes it, and its
/// layout noticed ([`LegacyLayout`], announced once per device): one Metal may keep losslessly
/// compressed, whether its bytes are Metal's (`RGBA8Unorm`, a 64x64 render target, 16x16 read
/// textures) or not (an `R32Uint` or `RGBA8Uint` render target of 16x16 or more, a 3D
/// `RGBA8Unorm` render target, an extended-range `BGRA10_XR` texture: measured, their bytes
/// copied out or in differing from Metal's on MoltenVK); and a 4-sample render target without
/// the property. A depth or stencil texture with `allowGPUOptimizedContents`, whose bytes are
/// not Metal's at any shape measured (`Depth16Unorm`, `Depth32Float` and `Stencil8`, 8x8 and
/// 64x64), dies at the placement, naming the heap and the reason
#[test]
fn legacy_heap_textures_metal_may_compress_are_announced() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let d = dev();
    let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let plain = (4, (1, 1));
    let moltenvk = crate::gpu::resource::on_moltenvk(&d);
    let noticed = if moltenvk {
        LegacyLayout::Compressible
    } else {
        LegacyLayout::Driver
    };
    let taken =
        || crate::gpu::resource::LEGACY_LAYOUTS.with(|l| std::mem::take(&mut *l.borrow_mut()));
    for (case, shape, differs) in [
        (
            "RGBA8Unorm render target",
            LegacyShape::flat(70, plain, (64, 64), 0x5, None),
            false,
        ),
        (
            "RGBA8Unorm read",
            LegacyShape::flat(70, plain, (16, 16), 0x1, Some(true)),
            false,
        ),
        (
            "R32Uint render target",
            LegacyShape::flat(53, plain, (16, 16), 0x5, None),
            true,
        ),
        (
            "RGBA8Uint render target",
            LegacyShape::flat(73, plain, (64, 64), 0x4, None),
            true,
        ),
        (
            "BGRA10_XR read",
            LegacyShape::flat(552, (8, (1, 1)), (64, 64), 0x1, None),
            true,
        ),
        (
            "3D RGBA8Unorm render target",
            LegacyShape {
                texture_type: 7,
                depth: 8,
                ..LegacyShape::flat(70, plain, (16, 16), 0x5, None)
            },
            true,
        ),
    ] {
        let theirs = legacy_heap_bytes_on::<MetalPlatform>(&metal, &metal_desc, shape, 0x10000);
        let theirs = theirs.unwrap_or_else(|| panic!("Metal makes it: {case}"));
        taken();
        let made = legacy_heap_death(&d, shape).unwrap();
        let ours = made.unwrap_or_else(|line| panic!("{case}: {line}"));
        let ours = ours.unwrap_or_else(|| panic!("made: {case}"));
        assert_eq!(taken(), [noticed], "{case}");
        if moltenvk && differs {
            assert!(ours.1 != theirs.1 || ours.2 != theirs.2, "differs: {case}");
        }
    }
    let depth = if moltenvk {
        "depth or stencil"
    } else {
        "its own layout"
    };
    for (case, shape) in [
        (
            "Depth16Unorm",
            LegacyShape::flat(250, (2, (1, 1)), (8, 8), 0x1, Some(true)),
        ),
        (
            "Depth32Float",
            LegacyShape::flat(252, plain, (64, 64), 0x5, None),
        ),
        (
            "Stencil8",
            LegacyShape::flat(253, (1, (1, 1)), (64, 64), 0x1, Some(true)),
        ),
    ] {
        assert!(
            legacy_heap_bytes_on::<MetalPlatform>(&metal, &metal_desc, shape, 0x10000).is_some(),
            "Metal makes it: {case}"
        );
        let line = legacy_heap_death(&d, shape).unwrap().unwrap_err();
        assert!(line.contains(depth), "{case}: {line}");
    }
    let multisample = TextureDesc {
        texture_type: 4,
        pixel_format: 70,
        width: 8,
        height: 8,
        sample_count: 4,
        usage: 0x4,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let metal_heap = heap_of::<MetalPlatform>(&metal, 0x10000, &options)
        .heap
        .unwrap();
    let made = metal_heap
        .new_texture(&metal_desc(&multisample), 0)
        .unwrap();
    assert!(made.is_some(), "Metal makes the multisample texture");
    let heap = heap_of::<VulkanPlatform>(&d, 0x10000, &options)
        .heap
        .unwrap();
    let _cover = heap
        .new_buffer_of_length(0x10000, 0x20, 0)
        .unwrap()
        .unwrap();
    let desc = platform().texture_descriptor(&multisample);
    taken();
    let placed = heap.new_texture(&desc, 0).unwrap();
    assert!(placed.is_some(), "the multisample texture");
    let noticed = if moltenvk {
        LegacyLayout::Multisample
    } else {
        LegacyLayout::Driver
    };
    assert_eq!(taken(), [noticed], "the multisample texture");
}
