//! The encoder's state: barriers and fences, fixed-function state, the vertex, fragment and tile
//! bindings, the SPI state setters, and the tile-dimensions reply (0xa4).
//!
//! - Every handler reads its record first, then checks it against the device's limits, then looks
//!   objects up, then sends, in that order (`tests::set_vertex_sampler_state`).
//! - A list setter is bounded first by the end of its range against the device's limit (two
//!   `u32` halves, so the sum cannot wrap); each entry is then read and looked up before the
//!   next is read, so a short list fails after the lookups before it (`tests::set_fragment_buffers`).
//! - 0x66 and 0x67 bound the attachment index with `>`, 0x96 with `>=`
//!   (`tests::set_color_store_action`, `tests::set_color_resolve_texture`).
//! - A render pipeline state found as none is sent and kept, so the next draw fails; the same
//!   pipeline again is neither released nor retained (`tests::set_render_pipeline_state`).
//! - 0xa4 alone checks the encoder, before it asks for the reply cursor; the reply is the tile
//!   width, then its height, each cut to `u16`, written at decode time
//!   (`tests::get_tile_dimensions`).

use zerocopy::{Immutable, IntoBytes};

use super::{RaiseKind, RenderDecoder, refs};
use crate::deserializer::records::{
    CmdGetTileDimensions, CmdRenderMemoryBarrierScope, CmdRenderUpdateFence, CmdRenderWaitForFence,
    CmdSetAlphaTestReferenceValue, CmdSetBlendColor, CmdSetBufferEntry,
    CmdSetBufferEntryWithStride, CmdSetBufferOffset, CmdSetBufferOffsetWithStride, CmdSetBuffers,
    CmdSetBuffersWithStride, CmdSetClipPlane, CmdSetColorResolveTexture, CmdSetDepthBias,
    CmdSetDepthStencilResolveTexture, CmdSetDepthStencilState, CmdSetFloatValue, CmdSetLineWidth,
    CmdSetPointSize, CmdSetPrimitiveRestartEnabledIndex, CmdSetProvokingVertexMode,
    CmdSetRenderPipelineState, CmdSetSamplerState, CmdSetSamplerStates,
    CmdSetSamplerStatesLODClamp, CmdSetSamplerStatesLODClampEntry, CmdSetScissorRect,
    CmdSetScissorRectEntry, CmdSetScissorRects, CmdSetStencilRef, CmdSetStoreAction,
    CmdSetStoreActionOption, CmdSetTesselationFactorBuffer, CmdSetTextures,
    CmdSetTransformFeedbackState, CmdSetTriangleFillModeFrontBack, CmdSetValue,
    CmdSetVertexAmplificationCount, CmdSetVertexAmplificationMode, CmdSetViewport,
    CmdSetViewportEntry, CmdSetViewportTransformEnabled, CmdSetViewports,
    CmdSetVisibiliyResultMode, ReplyTileDimensions,
    VertexAmplificationViewMapping as ViewMappingRecord,
};
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::num::{ToUsize, Truncate};
use crate::platform::render::{RenderCmd, RenderEncoder as _};
use crate::platform::types::{
    MtlScissorRect, MtlViewport, NsRange, VertexAmplificationViewMapping,
};
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::rt::Hold;

/// The range `location`, `length` of a list setter of `what` (a plural noun), refused with
/// `kind` when its end is past `limit`. Both halves are `u32`, so the sum cannot wrap.
fn bounded_range(
    location: u32,
    length: u32,
    limit: u64,
    kind: RaiseKind,
    what: &'static str,
) -> RaiseResult<NsRange> {
    let range = NsRange::new(u64::from(location), u64::from(length));
    if range.location.wrapping_add(range.length) > limit {
        crate::raise!(
            kind,
            format_args!("{what} {location}+{length} are past the device's {limit}")
        );
    }
    Ok(range)
}

/// Writes `reply` at the cursor, which moves past it.
fn write_reply<T: IntoBytes + Immutable>(
    writer: &mut WritableCursor,
    reply: &T,
) -> RaiseResult<()> {
    match writer.write(reply) {
        Ok(()) => Ok(()),
        Err(error) => Err(Raise::new(RaiseKind::ReplyWrite)
            .with_detail(format_args!("reply not writable: {error}"))),
    }
}

/// The buffers and offsets of a buffer list.
type BufferList<P> = (Vec<Option<<P as Platform>::Buffer>>, Vec<u64>);

/// The samplers and the two clamp arrays of a LOD-clamp sampler list.
type SamplerLodList<P> = (
    Vec<Option<<P as Platform>::SamplerState>>,
    Vec<f32>,
    Vec<f32>,
);

impl<P: Platform> RenderDecoder<P> {
    // ------------------------------------------------------------------ list readers

    /// `count` buffer entries, each read then looked up.
    fn buffer_entries(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
        count: u32,
    ) -> RaiseResult<BufferList<P>> {
        let mut buffers = Vec::with_capacity(count.to_usize());
        let mut offsets = Vec::with_capacity(count.to_usize());
        for _ in 0..count {
            let entry: CmdSetBufferEntry = c.read_or_raise(OnFault::Return)?;
            buffers.push(self.buffer(hold, entry.buffer_reference)?);
            offsets.push(entry.offset);
        }
        Ok((buffers, offsets))
    }

    /// `count` sampler references, each read then looked up.
    fn sampler_references(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
        count: u32,
    ) -> RaiseResult<Vec<Option<P::SamplerState>>> {
        let mut samplers = Vec::with_capacity(count.to_usize());
        for _ in 0..count {
            let reference: u32 = c.read_or_raise(OnFault::Return)?;
            samplers.push(self.sampler_state(hold, reference)?);
        }
        Ok(samplers)
    }

    /// `count` LOD-clamp sampler entries, each read then looked up.
    fn sampler_lod_entries(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
        count: u32,
    ) -> RaiseResult<SamplerLodList<P>> {
        let n = count.to_usize();
        let (mut samplers, mut mins, mut maxs) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for _ in 0..count {
            let entry: CmdSetSamplerStatesLODClampEntry = c.read_or_raise(OnFault::Return)?;
            samplers.push(self.sampler_state(hold, entry.sampler_state_reference)?);
            mins.push(entry.lod_min_clamp);
            maxs.push(entry.lod_max_clamp);
        }
        Ok((samplers, mins, maxs))
    }

    /// `count` texture references, each read then looked up.
    fn texture_references(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
        count: u32,
    ) -> RaiseResult<Vec<Option<P::Texture>>> {
        let mut textures = Vec::with_capacity(count.to_usize());
        for _ in 0..count {
            let reference: u32 = c.read_or_raise(OnFault::Return)?;
            textures.push(self.texture(hold, reference)?);
        }
        Ok(textures)
    }

    // ------------------------------------------------------------------ barriers and fences

    /// 0x16. Reads nothing and always raises: the barrier is unsupported.
    pub(super) fn barrier_resources(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let _ = c;
        crate::raise!(
            RaiseKind::BarrierResourcesUnsupported,
            "resource barriers are not supported"
        )
    }

    /// 0x17.
    pub(super) fn barrier_scope(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdRenderMemoryBarrierScope = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::MemoryBarrierWithScope {
            scope: u64::from(command.scope),
            after: u64::from(command.after_stages),
            before: u64::from(command.before_stages),
        })
    }

    /// 0x18: the fence is required.
    pub(super) fn update_fence(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdRenderUpdateFence = c.read_or_raise(OnFault::Return)?;
        let fence = self
            .deserializer()
            .require_fence(hold, command.fence_reference)?;
        self.send(RenderCmd::UpdateFence {
            fence: &fence,
            stages: u64::from(command.after_stages),
        })
    }

    /// 0x19: the fence is required.
    pub(super) fn wait_for_fence(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdRenderWaitForFence = c.read_or_raise(OnFault::Return)?;
        let fence = self
            .deserializer()
            .require_fence(hold, command.fence_reference)?;
        self.send(RenderCmd::WaitForFence {
            fence: &fence,
            stages: u64::from(command.before_stages),
        })
    }

    // ------------------------------------------------------------------ fixed-function state

    /// 0x65.
    pub(super) fn set_blend_color(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBlendColor = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetBlendColorRed {
            red: command.red,
            green: command.green,
            blue: command.blue,
            alpha: command.alpha,
        })
    }

    /// 0x66.
    pub(super) fn set_color_store_action(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetStoreAction = c.read_or_raise(OnFault::Return)?;
        // 0x66 compares with `>` (0x96 with `>=`).
        if command.color_attachment_index > self.max_color_attachment_index {
            crate::raise!(
                RaiseKind::RenderTargetIndex,
                format_args!(
                    "colour attachment index {} is past the device's limit",
                    command.color_attachment_index
                )
            );
        }
        self.send(RenderCmd::SetColorStoreAction {
            store_action: u64::from(command.store_action),
            color_attachment_index: u64::from(command.color_attachment_index),
        })
    }

    /// 0x67.
    pub(super) fn set_color_store_action_options(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetStoreActionOption = c.read_or_raise(OnFault::Return)?;
        // 0x67 compares with `>`.
        if command.color_attachment_index > self.max_color_attachment_index {
            crate::raise!(
                RaiseKind::RenderTargetIndex,
                format_args!("colour attachment index {} is past the device's limit", {
                    command.color_attachment_index
                })
            );
        }
        self.send(RenderCmd::SetColorStoreActionOptions {
            store_action_options: command.store_action_options,
            color_attachment_index: u64::from(command.color_attachment_index),
        })
    }

    /// 0x68: reference 0 raises before any lookup; a state found as none is sent as none.
    pub(super) fn set_depth_stencil_state(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetDepthStencilState = c.read_or_raise(OnFault::Return)?;
        if command.depth_stencil_state_reference == 0 {
            crate::raise!(
                RaiseKind::DepthStencilStateReference,
                "depth-stencil state reference 0"
            );
        }
        let state = self
            .deserializer()
            .object_delegate()
            .get_depth_stencil_state(hold, command.depth_stencil_state_reference)?;
        self.send(RenderCmd::SetDepthStencilState {
            depth_stencil_state: state.as_ref(),
        })
    }

    /// 0x69. The dispatch sends both 0x69 and 0x6a here, so 0x6a decodes as SetDepthStoreAction.
    pub(super) fn set_depth_store_action(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetDepthStoreAction {
            store_action: command.value,
        })
    }

    /// 0x6b.
    pub(super) fn set_cull_mode(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetCullMode {
            cull_mode: command.value,
        })
    }

    /// 0x6c.
    pub(super) fn set_depth_bias(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetDepthBias = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetDepthBias {
            depth_bias: command.depth_bias,
            slope_scale: command.slope_scale,
            clamp: command.clamp,
        })
    }

    /// 0x6d: an SPI; an encoder without it raises.
    pub(super) fn set_depth_clip_mode(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetDepthClipModeSpi {
            mode: command.value,
        })
    }

    // ------------------------------------------------------------------ fragment bindings

    /// 0x6e.
    pub(super) fn set_fragment_buffers(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetBuffers = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_fragment_buffers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::BufferCount,
            "buffers",
        )?;
        let (buffers, offsets) = self.buffer_entries(hold, c, command.range_length)?;
        self.send(RenderCmd::SetFragmentBuffers {
            buffers: &refs(&buffers),
            offsets: &offsets,
            range,
        })
    }

    /// 0x6f.
    pub(super) fn set_fragment_buffer_offset(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBufferOffset = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_fragment_buffers) {
            crate::raise!(
                RaiseKind::BufferIndex,
                format_args!("buffer index {index} is past the device's limit")
            );
        }
        self.send(RenderCmd::SetFragmentBufferOffset {
            offset: command.offset,
            index,
        })
    }

    /// 0x70.
    pub(super) fn set_fragment_sampler_states(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStates = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_fragment_samplers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::SamplerCount,
            "samplers",
        )?;
        let samplers = self.sampler_references(hold, c, command.range_length)?;
        self.send(RenderCmd::SetFragmentSamplerStates {
            samplers: &refs(&samplers),
            range,
        })
    }

    /// 0x71.
    pub(super) fn set_fragment_sampler_states_lod_clamp(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStatesLODClamp = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_fragment_samplers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::SamplerCount,
            "samplers",
        )?;
        let (samplers, mins, maxs) = self.sampler_lod_entries(hold, c, command.range_length)?;
        self.send(RenderCmd::SetFragmentSamplerStatesLodClamps {
            samplers: &refs(&samplers),
            lod_min_clamps: &mins,
            lod_max_clamps: &maxs,
            range,
        })
    }

    /// 0x72.
    pub(super) fn set_fragment_textures(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetTextures = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_fragment_textures);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::TextureCount,
            "textures",
        )?;
        let textures = self.texture_references(hold, c, command.range_length)?;
        self.send(RenderCmd::SetFragmentTextures {
            textures: &refs(&textures),
            range,
        })
    }

    /// 0x73.
    pub(super) fn set_front_facing_winding(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetFrontFacingWinding {
            front_facing_winding: command.value,
        })
    }

    /// 0x74: reference 0 raises before any lookup. The state found, none included, is sent,
    /// then kept unless it is the one already kept.
    pub(super) fn set_render_pipeline_state(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetRenderPipelineState = c.read_or_raise(OnFault::Return)?;
        if command.pipeline_state_reference == 0 {
            crate::raise!(
                RaiseKind::RenderPipelineReference,
                "render pipeline state reference 0"
            );
        }
        let pipeline = self
            .deserializer()
            .object_delegate()
            .get_render_pipeline_state(hold, command.pipeline_state_reference)?;
        self.send(RenderCmd::SetRenderPipelineState {
            pipeline_state: pipeline.as_ref(),
        })?;
        // By identity: the same pipeline is neither released nor retained.
        if self.pipeline.as_ref().map(Handle::id) != pipeline.as_ref().map(Handle::id) {
            self.pipeline = pipeline;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ scissors, stencil

    /// 0x75.
    pub(super) fn set_scissor_rect(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetScissorRect = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetScissorRect {
            rect: MtlScissorRect {
                x: command.x,
                y: command.y,
                width: command.width,
                height: command.height,
            },
        })
    }

    /// 0x76: a `u64` count bounded by the device's `maxViewportCount`.
    pub(super) fn set_scissor_rects(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetScissorRects = c.read_or_raise(OnFault::Return)?;
        if command.count > self.device_limit(|c| c.limits.max_viewport_count) {
            crate::raise!(
                RaiseKind::ViewportCount,
                format_args!("{} viewports, past the device's limit", command.count)
            );
        }
        let mut rects = Vec::with_capacity(command.count.to_usize());
        for _ in 0..command.count {
            let entry: CmdSetScissorRectEntry = c.read_or_raise(OnFault::Return)?;
            rects.push(MtlScissorRect {
                x: entry.x,
                y: entry.y,
                width: entry.width,
                height: entry.height,
            });
        }
        self.send(RenderCmd::SetScissorRects {
            scissor_rects: &rects,
        })
    }

    /// 0x77.
    pub(super) fn set_stencil_ref(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetStencilRef = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetStencilFrontReferenceValue {
            front_reference_value: command.front_reference_value,
            back_reference_value: command.back_reference_value,
        })
    }

    /// 0x78.
    pub(super) fn set_stencil_store_action(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetStencilStoreAction {
            store_action: command.value,
        })
    }

    /// 0x79.
    pub(super) fn set_stencil_store_action_options(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetStencilStoreActionOptions {
            store_action_options: command.value,
        })
    }

    // ------------------------------------------------------------------ tessellation, fill

    /// 0x7a: the buffer found, none included.
    pub(super) fn set_tessellation_factor_buffer(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetTesselationFactorBuffer = c.read_or_raise(OnFault::Return)?;
        let buffer = self.buffer(hold, command.buffer_reference)?;
        self.send(RenderCmd::SetTessellationFactorBuffer {
            buffer: buffer.as_ref(),
            offset: command.offset,
            instance_stride: command.instance_stride,
        })
    }

    /// 0x7b.
    pub(super) fn set_tessellation_factor_scale(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetFloatValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetTessellationFactorScale {
            scale: command.value,
        })
    }

    /// 0x7c.
    pub(super) fn set_triangle_fill_mode(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetTriangleFillMode {
            fill_mode: command.value,
        })
    }

    // ------------------------------------------------------------------ vertex bindings

    /// 0x7d.
    pub(super) fn set_vertex_buffers(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBuffers = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_vertex_buffers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::BufferCount,
            "buffers",
        )?;
        let (buffers, offsets) = self.buffer_entries(hold, c, command.range_length)?;
        self.send(RenderCmd::SetVertexBuffers {
            buffers: &refs(&buffers),
            offsets: &offsets,
            range,
        })
    }

    /// 0xa5: every entry is read and looked up before the stride support is checked.
    pub(super) fn set_vertex_buffers_with_stride(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetBuffersWithStride = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_vertex_buffers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::BufferCount,
            "buffers",
        )?;
        let n = command.range_length.to_usize();
        let (mut buffers, mut offsets, mut strides) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for _ in 0..command.range_length {
            let entry: CmdSetBufferEntryWithStride = c.read_or_raise(OnFault::Return)?;
            buffers.push(self.buffer(hold, entry.buffer_reference)?);
            offsets.push(entry.offset);
            strides.push(entry.attribute_stride);
        }
        if !self.device_supports(|f| f.supports_dynamic_attribute_stride) {
            crate::raise!(
                RaiseKind::AttributeStrideUnsupported,
                "the device does not support dynamic attribute strides"
            );
        }
        self.send(RenderCmd::SetVertexBuffersAttributeStrides {
            buffers: &refs(&buffers),
            offsets: &offsets,
            strides: &strides,
            range,
        })
    }

    /// 0xa6: the index first, then the stride support.
    pub(super) fn set_vertex_buffer_offset_with_stride(
        &mut self,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetBufferOffsetWithStride = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_vertex_buffers) {
            crate::raise!(
                RaiseKind::BufferIndex,
                format_args!("buffer index {index} is past the device's limit")
            );
        }
        if !self.device_supports(|f| f.supports_dynamic_attribute_stride) {
            crate::raise!(
                RaiseKind::AttributeStrideUnsupported,
                "the device does not support dynamic attribute strides"
            );
        }
        self.send(RenderCmd::SetVertexBufferOffsetAttributeStride {
            offset: command.offset,
            stride: command.attribute_stride,
            index,
        })
    }

    /// 0x7e.
    pub(super) fn set_vertex_buffer_offset(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBufferOffset = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_vertex_buffers) {
            crate::raise!(
                RaiseKind::BufferIndex,
                format_args!("buffer index {index} is past the device's limit")
            );
        }
        self.send(RenderCmd::SetVertexBufferOffset {
            offset: command.offset,
            index,
        })
    }

    /// 0x7f.
    pub(super) fn set_vertex_sampler_states(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStates = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_vertex_samplers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::SamplerCount,
            "samplers",
        )?;
        let samplers = self.sampler_references(hold, c, command.range_length)?;
        self.send(RenderCmd::SetVertexSamplerStates {
            samplers: &refs(&samplers),
            range,
        })
    }

    /// 0x80.
    pub(super) fn set_vertex_sampler_states_lod_clamp(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStatesLODClamp = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_vertex_samplers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::SamplerCount,
            "samplers",
        )?;
        let (samplers, mins, maxs) = self.sampler_lod_entries(hold, c, command.range_length)?;
        self.send(RenderCmd::SetVertexSamplerStatesLodClamps {
            samplers: &refs(&samplers),
            lod_min_clamps: &mins,
            lod_max_clamps: &maxs,
            range,
        })
    }

    /// 0x81.
    pub(super) fn set_vertex_textures(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetTextures = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_vertex_textures);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::TextureCount,
            "textures",
        )?;
        let textures = self.texture_references(hold, c, command.range_length)?;
        self.send(RenderCmd::SetVertexTextures {
            textures: &refs(&textures),
            range,
        })
    }

    // ------------------------------------------------------------------ viewports

    /// 0x82.
    pub(super) fn set_viewport(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetViewport = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetViewport {
            viewport: MtlViewport {
                origin_x: command.origin_x,
                origin_y: command.origin_y,
                width: command.width,
                height: command.height,
                znear: command.znear,
                zfar: command.zfar,
            },
        })
    }

    /// 0x83.
    pub(super) fn set_viewports(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetViewports = c.read_or_raise(OnFault::Return)?;
        if u64::from(command.count) > self.device_limit(|c| c.limits.max_viewport_count) {
            crate::raise!(
                RaiseKind::ViewportCount,
                format_args!("{} viewports, past the device's limit", command.count)
            );
        }
        let mut viewports = Vec::with_capacity(command.count.to_usize());
        for _ in 0..command.count {
            let entry: CmdSetViewportEntry = c.read_or_raise(OnFault::Return)?;
            viewports.push(MtlViewport {
                origin_x: entry.origin_x,
                origin_y: entry.origin_y,
                width: entry.width,
                height: entry.height,
                znear: entry.znear,
                zfar: entry.zfar,
            });
        }
        self.send(RenderCmd::SetViewports {
            viewports: &viewports,
        })
    }

    /// 0x84: the record holds the offset first, then the mode.
    pub(super) fn set_visibility_result_mode(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetVisibiliyResultMode = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetVisibilityResultMode {
            mode: command.mode,
            offset: command.offset,
        })
    }

    // ------------------------------------------------------------------ SPI state

    /// 0x88.
    pub(super) fn set_line_width(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetLineWidth = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetLineWidth {
            line_width: command.line_width,
        })
    }

    /// 0x8a.
    pub(super) fn set_alpha_test_reference_value(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetAlphaTestReferenceValue = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetAlphaTestReferenceValue {
            reference_value: command.reference_value,
        })
    }

    /// 0x8b.
    pub(super) fn set_point_size(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetPointSize = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetPointSize {
            point_size: command.point_size,
        })
    }

    /// 0x8c: six clip planes; the host's encoder has no such selector
    /// ([`Self::send_unrecognized`]).
    pub(super) fn set_clip_plane(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetClipPlane = c.read_or_raise(OnFault::Return)?;
        if command.index >= 6 {
            crate::raise!(
                RaiseKind::ClipPlaneIndex,
                format_args!("clip plane index {} is past 5", command.index)
            );
        }
        self.send_unrecognized("setClipPlane:p2:p3:p4:atIndex:")
    }

    /// 0x8d. The index is checked before the sampler lookup; the host's encoder has no such
    /// selector ([`Self::send_unrecognized`]).
    pub(super) fn set_vertex_sampler_state(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerState = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_vertex_samplers) {
            crate::raise!(
                RaiseKind::SamplerIndex,
                format_args!("sampler index {index} is past the device's limit")
            );
        }
        let sampler = self.sampler_state(hold, command.sampler_state_reference)?;
        let sent = self
            .send_unrecognized("setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:");
        drop(sampler);
        sent
    }

    /// 0x8e. Bounded by `maxVertexSamplers`, not the fragment limit. The index is checked before
    /// the sampler lookup; the host's encoder has no such selector
    /// ([`Self::send_unrecognized`]).
    pub(super) fn set_fragment_sampler_state(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerState = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_vertex_samplers) {
            crate::raise!(
                RaiseKind::SamplerIndex,
                format_args!("sampler index {index} is past the device's limit")
            );
        }
        let sampler = self.sampler_state(hold, command.sampler_state_reference)?;
        let sent = self
            .send_unrecognized("setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:");
        drop(sampler);
        sent
    }

    /// 0x8f: any non-zero word is enabled.
    pub(super) fn set_viewport_transform_enabled(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetViewportTransformEnabled = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetViewportTransformEnabled {
            enabled: command.enabled != 0,
        })
    }

    /// 0x90.
    pub(super) fn set_provoking_vertex_mode(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetProvokingVertexMode = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetProvokingVertexMode {
            mode: u64::from(command.mode),
        })
    }

    /// 0x91. The index is the strips' restart index, passed on whatever it is. Apple's
    /// deserializer bounds it by `maxViewportCount` (read, docs/testing.md sections 6.16 and
    /// 7.1), but the guest's OpenGL ES renderer (AppleMetalGLRenderer) sends 0xffffffff with its
    /// draws, so that bound faulted every OpenGL ES draw; this port does not keep it.
    pub(super) fn set_primitive_restart_index_enabled(
        &mut self,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetPrimitiveRestartEnabledIndex = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetPrimitiveRestartEnabled {
            enabled: command.enabled != 0,
            index: u64::from(command.index),
        })
    }

    /// 0x92: the host's encoder has no such selector ([`Self::send_unrecognized`]).
    pub(super) fn set_triangle_fill_mode_front_back(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let _: CmdSetTriangleFillModeFrontBack = c.read_or_raise(OnFault::Return)?;
        self.send_unrecognized("setTriangleFrontFillMode:backFillMode:")
    }

    /// 0x93: the host's encoder has no such selector ([`Self::send_unrecognized`]).
    pub(super) fn set_transform_feedback_state(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let _: CmdSetTransformFeedbackState = c.read_or_raise(OnFault::Return)?;
        self.send_unrecognized("setTransformFeedbackState:")
    }

    /// The send of a render-encoder selector the host's encoder does not implement (measured on
    /// macOS 26.6: `class_getInstanceMethod` answers NULL for each of the five records that
    /// call this). Metal answers such a send with `NSInvalidArgumentException`, "unrecognized
    /// selector", which the decoder's catch turns into a faulted exec; this raises the same,
    /// named by the selector, without sending it. With no encoder the send is a message to nil,
    /// and nothing happens. It comes after the record's checks and lookups, as the send did.
    fn send_unrecognized(&self, selector: &'static str) -> RaiseResult<()> {
        match &self.encoder {
            Some(_) => Err(Raise::foreign(selector).with_detail("unrecognized selector")),
            None => Ok(()),
        }
    }

    /// 0x94: reads nothing.
    pub(super) fn set_depth_cleared(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let _ = c;
        self.send(RenderCmd::SetDepthCleared)
    }

    /// 0x95: reads nothing.
    pub(super) fn set_stencil_cleared(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let _ = c;
        self.send(RenderCmd::SetStencilCleared)
    }

    /// 0x96. Compares with `>=` against the highest colour index, so index 0 fails when only
    /// attachment 0 has a texture.
    pub(super) fn set_color_resolve_texture(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetColorResolveTexture = c.read_or_raise(OnFault::Return)?;
        if u32::from(command.color_attachment_index) >= self.max_color_attachment_index {
            crate::raise!(
                RaiseKind::AttachmentIndex,
                format_args!(
                    "colour attachment index {} is past the device's limit",
                    command.color_attachment_index
                )
            );
        }
        let texture = self.texture(hold, command.texture_reference)?;
        self.send(RenderCmd::SetColorResolveTexture {
            texture: texture.as_ref(),
            slice: u64::from(command.slice),
            depth_plane: u64::from(command.depth_plane),
            level: u64::from(command.level),
            y_invert: command.y_invert != 0,
            index: u64::from(command.color_attachment_index),
        })
    }

    /// 0x97.
    pub(super) fn set_depth_resolve_texture(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetDepthStencilResolveTexture = c.read_or_raise(OnFault::Return)?;
        let texture = self.texture(hold, command.texture_reference)?;
        self.send(RenderCmd::SetDepthResolveTexture {
            texture: texture.as_ref(),
            slice: u64::from(command.slice),
            depth_plane: u64::from(command.depth_plane),
            level: u64::from(command.level),
            y_invert: command.y_invert != 0,
        })
    }

    /// 0x98.
    pub(super) fn set_stencil_resolve_texture(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetDepthStencilResolveTexture = c.read_or_raise(OnFault::Return)?;
        let texture = self.texture(hold, command.texture_reference)?;
        self.send(RenderCmd::SetStencilResolveTexture {
            texture: texture.as_ref(),
            slice: u64::from(command.slice),
            depth_plane: u64::from(command.depth_plane),
            level: u64::from(command.level),
            y_invert: command.y_invert != 0,
        })
    }

    /// 0x99.
    pub(super) fn set_vertex_amplification_mode(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetVertexAmplificationMode = c.read_or_raise(OnFault::Return)?;
        self.send(RenderCmd::SetVertexAmplificationMode {
            mode: u64::from(command.mode),
            value: u64::from(command.value),
        })
    }

    /// 0x9a: the count is bounded by the device's `maxVertexAmplificationCount`.
    pub(super) fn set_vertex_amplification_count(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetVertexAmplificationCount = c.read_or_raise(OnFault::Return)?;
        if u64::from(command.count) > self.device_limit(|c| c.max_vertex_amplification_count) {
            crate::raise!(
                RaiseKind::VertexAmplificationCount,
                format_args!(
                    "vertex amplification count {} is past the device's limit",
                    command.count
                )
            );
        }
        let mut view_mappings = Vec::with_capacity(command.count.to_usize());
        for _ in 0..command.count {
            let mapping: ViewMappingRecord = c.read_or_raise(OnFault::Return)?;
            view_mappings.push(VertexAmplificationViewMapping {
                viewport_array_index_offset: mapping.viewport_array_index_offset,
                render_target_array_index_offset: mapping.render_target_array_index_offset,
            });
        }
        self.send(RenderCmd::SetVertexAmplificationCount {
            view_mappings: &view_mappings,
        })
    }

    // ------------------------------------------------------------------ tile bindings

    /// 0x9d.
    pub(super) fn set_tile_buffers(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBuffers = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_tile_buffers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::BufferCount,
            "buffers",
        )?;
        let (buffers, offsets) = self.buffer_entries(hold, c, command.range_length)?;
        self.send(RenderCmd::SetTileBuffers {
            buffers: &refs(&buffers),
            offsets: &offsets,
            range,
        })
    }

    /// 0x9e.
    pub(super) fn set_tile_buffer_offset(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBufferOffset = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_tile_buffers) {
            crate::raise!(
                RaiseKind::BufferIndex,
                format_args!("buffer index {index} is past the device's limit")
            );
        }
        self.send(RenderCmd::SetTileBufferOffset {
            offset: command.offset,
            index,
        })
    }

    /// 0x9f.
    pub(super) fn set_tile_sampler_states(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStates = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_tile_samplers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::SamplerCount,
            "samplers",
        )?;
        let samplers = self.sampler_references(hold, c, command.range_length)?;
        self.send(RenderCmd::SetTileSamplerStates {
            samplers: &refs(&samplers),
            range,
        })
    }

    /// 0xa0.
    pub(super) fn set_tile_sampler_states_lod_clamp(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStatesLODClamp = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_tile_samplers);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::SamplerCount,
            "samplers",
        )?;
        let (samplers, mins, maxs) = self.sampler_lod_entries(hold, c, command.range_length)?;
        self.send(RenderCmd::SetTileSamplerStatesLodClamps {
            samplers: &refs(&samplers),
            lod_min_clamps: &mins,
            lod_max_clamps: &maxs,
            range,
        })
    }

    /// 0xa1.
    pub(super) fn set_tile_textures(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetTextures = c.read_or_raise(OnFault::Return)?;
        let limit = self.device_limit(|c| c.limits.max_tile_textures);
        let range = bounded_range(
            command.range_location,
            command.range_length,
            limit,
            RaiseKind::TextureCount,
            "textures",
        )?;
        let textures = self.texture_references(hold, c, command.range_length)?;
        self.send(RenderCmd::SetTileTextures {
            textures: &refs(&textures),
            range,
        })
    }

    /// 0xa4: the one handler that checks the encoder. The reply cursor moves to the offset
    /// (clamped at its end), then takes the width, then the height.
    pub(super) fn get_tile_dimensions(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdGetTileDimensions = c.read_or_raise(OnFault::Return)?;
        let Some(encoder) = self.encoder.as_ref() else {
            crate::raise!(
                RaiseKind::TileDimensionsWithoutEncoder,
                "tile dimensions asked with no render encoder"
            );
        };
        let Some(mut writer) = self
            .deserializer()
            .object_delegate()
            .new_memory_cursor_for_buffer(hold, command.reply_buffer_reference)?
        else {
            crate::raise!(
                RaiseKind::ReplyCursor,
                format_args!("no reply cursor for buffer {}", {
                    command.reply_buffer_reference
                })
            );
        };
        writer.0.advance(command.reply_buffer_offset);
        // The tile dimensions are cut to `u16`.
        let reply = ReplyTileDimensions {
            tile_width: encoder.tile_width().low_u16(),
            tile_height: encoder.tile_height().low_u16(),
        };
        write_reply(&mut writer, &reply)
    }
}
#[cfg(test)]
mod tests {

    use super::super::draw::test_support::*;
    use super::*;
    use crate::deserializer::records::VertexAmplificationViewMapping as ViewMappingRecord;

    use crate::platform::caps::DeviceCaps;
    use crate::raise::RaiseId;
    use crate::testkit::calllog::Arg;
    use crate::testkit::objects::kind;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Handler = fn(&mut RenderDecoder<FakePlatform>, &mut Cursor) -> RaiseResult<()>;

    /// A one-record, one-send handler: short fails its read with nothing sent; the
    /// record gives `sel(args)`; a nil encoder sends nothing and succeeds.
    fn simple(f: Handler, bytes: &[u8], sel: &str, args: Vec<Arg>) {
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&bytes[..bytes.len() - 1])));
        assert!(r.calls().is_empty());
        let mut r = Rig::new();
        f(&mut r.dec, &mut cursor(bytes)).unwrap();
        assert_eq!(r.encoder_calls(), vec![(sel_static(&r, sel), args)]);
        // A nil encoder: lookups and device getters still run; nothing is sent.
        let mut r = Rig::without_encoder();
        f(&mut r.dec, &mut cursor(bytes)).unwrap();
        assert!(r.encoder_calls().is_empty());
        assert!(!r.sels().contains(&sel));
    }

    /// The logged `&'static str` equal to `sel` (the log keeps the fake's literal).
    fn sel_static(r: &Rig, sel: &str) -> &'static str {
        r.sels()
            .into_iter()
            .find(|s| *s == sel)
            .unwrap_or_else(|| panic!("{sel} not sent: {:?}", r.sels()))
    }

    /// A handler whose selector the host's encoder does not implement: a short record fails its
    /// read; with an encoder the record raises what Metal raises for an unrecognized selector,
    /// named by `sel`, and sends nothing; with none, nothing happens.
    fn unrecognized(f: Handler, bytes: &[u8], sel: &'static str) {
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&bytes[..bytes.len() - 1])));
        let mut r = Rig::new();
        let e = f(&mut r.dec, &mut cursor(bytes)).unwrap_err();
        assert_eq!(e, Raise::foreign(sel).with_detail("unrecognized selector"));
        assert_eq!(e.id(), RaiseId::Platform);
        assert!(r.encoder_calls().is_empty());
        let mut r = Rig::without_encoder();
        f(&mut r.dec, &mut cursor(bytes)).unwrap();
        assert!(r.encoder_calls().is_empty());
    }

    fn value(v: u64) -> Vec<u8> {
        rec(&CmdSetValue { value: v })
    }

    // ------------------------------------------------------------------ barriers and fences

    #[test]
    fn render_barrier_resources() {
        // Always throws, reads nothing.
        {
            let mut r = Rig::new();
            let mut c = cursor(&[1, 2, 3, 4]);
            assert_raise(
                r.dec.barrier_resources(&mut c),
                RaiseKind::BarrierResourcesUnsupported,
                "resource barriers are not supported",
            );
            assert_eq!(c.remaining(), 4);
            assert!(r.calls().is_empty());
            let mut r = Rig::without_encoder();
            assert_raise(
                r.dec.barrier_resources(&mut cursor(&[])),
                RaiseKind::BarrierResourcesUnsupported,
                "resource barriers are not supported",
            );
        };
    }

    #[test]
    fn render_barrier_scope() {
        let cmd = CmdRenderMemoryBarrierScope {
            scope: 0xffff,
            after_stages: 0xff,
            before_stages: 1,
        };
        simple(
            RenderDecoder::barrier_scope,
            &rec(&cmd),
            "memoryBarrierWithScope:afterStages:beforeStages:",
            vec![Arg::U64(0xffff), Arg::U64(0xff), Arg::U64(1)],
        );
    }

    // ------------------------------------------------------------------ fixed-function state

    #[test]
    fn set_blend_color() {
        let cmd = CmdSetBlendColor {
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 1.0,
        };
        simple(
            RenderDecoder::set_blend_color,
            &rec(&cmd),
            "setBlendColorRed:green:blue:alpha:",
            vec![Arg::F32(0.25), Arg::F32(0.5), Arg::F32(0.75), Arg::F32(1.0)],
        );
    }

    #[test]
    fn set_color_store_action() {
        let f: Handler = RenderDecoder::set_color_store_action;
        let at = |i| {
            rec(&CmdSetStoreAction {
                store_action: 2,
                color_attachment_index: i,
            })
        };
        // `>`: with max 0, index 0 passes and 1 fails.
        simple(
            f,
            &at(0),
            "setColorStoreAction:atIndex:",
            vec![Arg::U64(2), Arg::U64(0)],
        );
        let mut r = Rig::new();
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(1))),
            RaiseKind::RenderTargetIndex,
            "colour attachment index 1 is past the device's limit",
        );
        assert!(r.calls().is_empty());
        let mut r = Rig::new();
        r.dec.max_color_attachment_index = 3;
        f(&mut r.dec, &mut cursor(&at(3))).unwrap();
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(4))),
            RaiseKind::RenderTargetIndex,
            "colour attachment index 4 is past the device's limit",
        );
    }

    #[test]
    fn set_color_store_action_options() {
        let f: Handler = RenderDecoder::set_color_store_action_options;
        let at = |i| {
            rec(&CmdSetStoreActionOption {
                store_action_options: u64::MAX,
                color_attachment_index: i,
            })
        };
        simple(
            f,
            &at(0),
            "setColorStoreActionOptions:atIndex:",
            vec![Arg::U64(u64::MAX), Arg::U64(0)],
        );
        let mut r = Rig::new();
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(1))),
            RaiseKind::RenderTargetIndex,
            "colour attachment index 1 is past the device's limit",
        );
        let mut r = Rig::new();
        r.dec.max_color_attachment_index = 7;
        f(&mut r.dec, &mut cursor(&at(7))).unwrap();
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(u32::MAX))),
            RaiseKind::RenderTargetIndex,
            "colour attachment index 4294967295 is past the device's limit",
        );
    }

    #[test]
    fn set_depth_store_action() {
        // The dispatch sends 0x69 and 0x6a both here.
        simple(
            RenderDecoder::set_depth_store_action,
            &value(0x1_0000_0002),
            "setDepthStoreAction:",
            vec![Arg::U64(0x1_0000_0002)],
        );
    }

    #[test]
    fn set_cull_mode() {
        simple(
            RenderDecoder::set_cull_mode,
            &value(2),
            "setCullMode:",
            vec![Arg::U64(2)],
        );
    }

    #[test]
    fn set_depth_bias() {
        let cmd = CmdSetDepthBias {
            depth_bias: -1.5,
            slope_scale: 2.0,
            clamp: 0.125,
        };
        simple(
            RenderDecoder::set_depth_bias,
            &rec(&cmd),
            "setDepthBias:slopeScale:clamp:",
            vec![Arg::F32(-1.5), Arg::F32(2.0), Arg::F32(0.125)],
        );
    }

    #[test]
    fn set_depth_clip_mode() {
        simple(
            RenderDecoder::set_depth_clip_mode,
            &value(1),
            "setDepthClipModeSPI:",
            vec![Arg::U64(1)],
        );
        // An encoder without the SPI raises; the exception goes on to the decoder's handler.
        {
            let mut r = Rig::new();
            r.p.script().push(
                "setDepthClipModeSPI:",
                Step::unrecognized("setDepthClipModeSPI:"),
            );
            let e = r
                .dec
                .set_depth_clip_mode(&mut cursor(&value(1)))
                .unwrap_err();
            assert_eq!(e.id(), RaiseId::Platform);
        };
    }

    #[test]
    fn set_front_facing_winding() {
        simple(
            RenderDecoder::set_front_facing_winding,
            &value(1),
            "setFrontFacingWinding:",
            vec![Arg::U64(1)],
        );
    }

    #[test]
    fn set_stencil_store_action() {
        simple(
            RenderDecoder::set_stencil_store_action,
            &value(3),
            "setStencilStoreAction:",
            vec![Arg::U64(3)],
        );
    }

    #[test]
    fn set_stencil_store_action_options() {
        simple(
            RenderDecoder::set_stencil_store_action_options,
            &value(1),
            "setStencilStoreActionOptions:",
            vec![Arg::U64(1)],
        );
    }

    #[test]
    fn set_triangle_fill_mode() {
        simple(
            RenderDecoder::set_triangle_fill_mode,
            &value(1),
            "setTriangleFillMode:",
            vec![Arg::U64(1)],
        );
    }

    #[test]
    fn set_stencil_ref() {
        let cmd = CmdSetStencilRef {
            front_reference_value: 1,
            back_reference_value: 0xffff_ffff,
        };
        simple(
            RenderDecoder::set_stencil_ref,
            &rec(&cmd),
            "setStencilFrontReferenceValue:backReferenceValue:",
            vec![Arg::U32(1), Arg::U32(0xffff_ffff)],
        );
    }

    #[test]
    fn set_tesselation_factor_scale() {
        simple(
            RenderDecoder::set_tessellation_factor_scale,
            &rec(&CmdSetFloatValue { value: 3.5 }),
            "setTessellationFactorScale:",
            vec![Arg::F32(3.5)],
        );
    }

    #[test]
    fn set_scissor_rect() {
        let cmd = CmdSetScissorRect {
            x: 1,
            y: 2,
            width: 3,
            height: 4,
        };
        simple(
            RenderDecoder::set_scissor_rect,
            &rec(&cmd),
            "setScissorRect:",
            vec![Arg::val(&MtlScissorRect {
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            })],
        );
    }

    #[test]
    fn set_scissor_rects() {
        let f: Handler = RenderDecoder::set_scissor_rects;
        let entry = |i: u64| CmdSetScissorRectEntry {
            x: i,
            y: i + 1,
            width: i + 2,
            height: i + 3,
        };
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&[0; 7])));
        // `uint64_t` count against maxViewportCount (16 by default).
        for count in [17u64, 1 << 32] {
            let mut r = Rig::new();
            assert_raise(
                f(&mut r.dec, &mut cursor(&rec(&CmdSetScissorRects { count }))),
                RaiseKind::ViewportCount,
                &format!("{count} viewports, past the device's limit"),
            );
            assert!(r.sels().is_empty());
        }
        // Short entries.
        let mut b = rec(&CmdSetScissorRects { count: 2 });
        b.extend_from_slice(&rec(&entry(0)));
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&b)));
        assert!(r.encoder_calls().is_empty());
        b.extend_from_slice(&rec(&entry(10)));
        let mut r = Rig::new();
        f(&mut r.dec, &mut cursor(&b)).unwrap();
        let rects = [
            MtlScissorRect {
                x: 0,
                y: 1,
                width: 2,
                height: 3,
            },
            MtlScissorRect {
                x: 10,
                y: 11,
                width: 12,
                height: 13,
            },
        ];
        assert_eq!(
            r.encoder_calls(),
            vec![("setScissorRects:count:", vec![Arg::val(&rects[..])])]
        );
        // Count 0 and count == limit pass.
        let mut r = Rig::new();
        f(
            &mut r.dec,
            &mut cursor(&rec(&CmdSetScissorRects { count: 0 })),
        )
        .unwrap();
        let mut r = Rig::new();
        r.set_caps(|c| c.limits.max_viewport_count = 1);
        let mut b = rec(&CmdSetScissorRects { count: 1 });
        b.extend_from_slice(&rec(&entry(0)));
        f(&mut r.dec, &mut cursor(&b)).unwrap();
    }

    #[test]
    fn set_viewport() {
        let cmd = CmdSetViewport {
            origin_x: 1.0,
            origin_y: 2.0,
            width: 3.0,
            height: 4.0,
            znear: 0.0,
            zfar: 1.0,
        };
        simple(
            RenderDecoder::set_viewport,
            &rec(&cmd),
            "setViewport:",
            vec![Arg::val(&MtlViewport {
                origin_x: 1.0,
                origin_y: 2.0,
                width: 3.0,
                height: 4.0,
                znear: 0.0,
                zfar: 1.0,
            })],
        );
    }

    #[test]
    fn set_viewports() {
        let f: Handler = RenderDecoder::set_viewports;
        let entry = |x: f64| CmdSetViewportEntry {
            origin_x: x,
            origin_y: 0.0,
            width: 8.0,
            height: 8.0,
            znear: 0.0,
            zfar: 1.0,
        };
        let vp = |x: f64| MtlViewport {
            origin_x: x,
            origin_y: 0.0,
            width: 8.0,
            height: 8.0,
            znear: 0.0,
            zfar: 1.0,
        };
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&[0; 3])));
        let mut r = Rig::new();
        r.set_caps(|c| c.limits.max_viewport_count = 2);
        assert_raise(
            f(&mut r.dec, &mut cursor(&rec(&CmdSetViewports { count: 3 }))),
            RaiseKind::ViewportCount,
            "3 viewports, past the device's limit",
        );
        let mut b = rec(&CmdSetViewports { count: 2 });
        b.extend_from_slice(&rec(&entry(1.0)));
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&b)));
        b.extend_from_slice(&rec(&entry(2.0)));
        let mut r = Rig::new();
        f(&mut r.dec, &mut cursor(&b)).unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "setViewports:count:",
                vec![Arg::val(&[vp(1.0), vp(2.0)][..])]
            )]
        );
    }

    #[test]
    fn set_visibility_result_mode() {
        // The mode is at +8, the offset at +0.
        let cmd = CmdSetVisibiliyResultMode { offset: 1, mode: 2 };
        simple(
            RenderDecoder::set_visibility_result_mode,
            &rec(&cmd),
            "setVisibilityResultMode:offset:",
            vec![Arg::U64(2), Arg::U64(1)],
        );
    }

    // ------------------------------------------------------------------ list setters

    /// An index-bounded offset setter: index == limit fails, limit - 1 passes.
    fn offset_case(
        f: Handler,
        limit: fn(&mut DeviceCaps) -> &mut u64,
        kind: RaiseKind,
        send: &'static str,
    ) {
        let at = |i| {
            rec(&CmdSetBufferOffset {
                index: i,
                offset: 0x40,
            })
        };
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&at(0)[..0xb])));
        let mut r = Rig::new();
        r.set_caps(|c| *limit(c) = 4);
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(4))),
            kind,
            "buffer index 4 is past the device's limit",
        );
        assert!(r.encoder_calls().is_empty());
        let mut r = Rig::new();
        r.set_caps(|c| *limit(c) = 4);
        f(&mut r.dec, &mut cursor(&at(3))).unwrap();
        assert_eq!(r.encoder_calls()[0].0, send);
        assert_eq!(r.encoder_calls()[0].1, vec![Arg::U64(0x40), Arg::U64(3)]);
    }

    #[test]
    fn set_fragment_buffer_offset() {
        offset_case(
            RenderDecoder::set_fragment_buffer_offset,
            |c| &mut c.limits.max_fragment_buffers,
            RaiseKind::BufferIndex,
            "setFragmentBufferOffset:atIndex:",
        );
    }

    #[test]
    fn set_vertex_buffer_offset() {
        offset_case(
            RenderDecoder::set_vertex_buffer_offset,
            |c| &mut c.limits.max_vertex_buffers,
            RaiseKind::BufferIndex,
            "setVertexBufferOffset:atIndex:",
        );
    }

    #[test]
    fn set_tile_buffer_offset() {
        offset_case(
            RenderDecoder::set_tile_buffer_offset,
            |c| &mut c.limits.max_tile_buffers,
            RaiseKind::BufferIndex,
            "setTileBufferOffset:atIndex:",
        );
    }

    #[test]
    fn set_vertex_buffer_offset_with_stride() {
        let f: Handler = RenderDecoder::set_vertex_buffer_offset_with_stride;
        let at = |i| {
            rec(&CmdSetBufferOffsetWithStride {
                index: i,
                offset: 0x40,
                attribute_stride: 12,
            })
        };
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&at(0)[..0x13])));
        // The index first, then stride support.
        let mut r = Rig::new();
        r.set_caps(|c| {
            c.limits.max_vertex_buffers = 4;
            c.flags.supports_dynamic_attribute_stride = false;
        });
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(4))),
            RaiseKind::BufferIndex,
            "buffer index 4 is past the device's limit",
        );
        assert!(r.sels().is_empty());
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(3))),
            RaiseKind::AttributeStrideUnsupported,
            "the device does not support dynamic attribute strides",
        );
        let mut r = Rig::new();
        f(&mut r.dec, &mut cursor(&at(3))).unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "setVertexBufferOffset:attributeStride:atIndex:",
                vec![Arg::U64(0x40), Arg::U64(12), Arg::U64(3)]
            )]
        );
    }

    // ------------------------------------------------------------------ SPI state

    #[test]
    fn set_line_width() {
        simple(
            RenderDecoder::set_line_width,
            &rec(&CmdSetLineWidth { line_width: 2.5 }),
            "setLineWidth:",
            vec![Arg::F32(2.5)],
        );
    }

    #[test]
    fn set_alpha_test_reference_value() {
        simple(
            RenderDecoder::set_alpha_test_reference_value,
            &rec(&CmdSetAlphaTestReferenceValue {
                reference_value: 0.5,
            }),
            "setAlphaTestReferenceValue:",
            vec![Arg::F32(0.5)],
        );
    }

    #[test]
    fn set_point_size() {
        simple(
            RenderDecoder::set_point_size,
            &rec(&CmdSetPointSize { point_size: 4.0 }),
            "setPointSize:",
            vec![Arg::F32(4.0)],
        );
    }

    #[test]
    fn set_clip_plane() {
        let f: Handler = RenderDecoder::set_clip_plane;
        let at = |i| {
            rec(&CmdSetClipPlane {
                p1: 1.0,
                p2: 2.0,
                p3: 3.0,
                p4: 4.0,
                index: i,
            })
        };
        unrecognized(f, &at(5), "setClipPlane:p2:p3:p4:atIndex:");
        // The index is checked first, with or without an encoder.
        {
            let mut r = Rig::new();
            assert_raise(
                f(&mut r.dec, &mut cursor(&at(6))),
                RaiseKind::ClipPlaneIndex,
                "clip plane index 6 is past 5",
            );
            assert!(r.calls().is_empty());
            let mut r = Rig::without_encoder();
            assert_raise(
                f(&mut r.dec, &mut cursor(&at(6))),
                RaiseKind::ClipPlaneIndex,
                "clip plane index 6 is past 5",
            );
        };
    }

    #[test]
    fn set_viewport_transform_enabled() {
        let f: Handler = RenderDecoder::set_viewport_transform_enabled;
        simple(
            f,
            &rec(&CmdSetViewportTransformEnabled { enabled: 0x100 }),
            "setViewportTransformEnabled:",
            vec![Arg::Bool(true)],
        );
        let mut r = Rig::new();
        f(
            &mut r.dec,
            &mut cursor(&rec(&CmdSetViewportTransformEnabled { enabled: 0 })),
        )
        .unwrap();
        assert_eq!(r.encoder_calls()[0].1, vec![Arg::Bool(false)]);
    }

    #[test]
    fn set_provoking_vertex_mode() {
        simple(
            RenderDecoder::set_provoking_vertex_mode,
            &rec(&CmdSetProvokingVertexMode { mode: 0xffff_ffff }),
            "setProvokingVertexMode:",
            vec![Arg::U64(0xffff_ffff)],
        );
    }

    #[test]
    fn set_primitive_restart_index_enabled() {
        let f: Handler = RenderDecoder::set_primitive_restart_index_enabled;
        let at = |enabled, index| rec(&CmdSetPrimitiveRestartEnabledIndex { enabled, index });
        simple(
            f,
            &at(2, 1),
            "setPrimitiveRestartEnabled:index:",
            vec![Arg::Bool(true), Arg::U64(1)],
        );
        // Not bounded by maxViewportCount, as Apple's is: AppleMetalGLRenderer's 0xffffffff.
        {
            let mut r = Rig::new();
            r.set_caps(|c| c.limits.max_viewport_count = 3);
            f(&mut r.dec, &mut cursor(&at(0, 0xffff_ffff))).unwrap();
            assert_eq!(
                r.encoder_calls()[0].1,
                vec![Arg::Bool(false), Arg::U64(0xffff_ffff)]
            );
        };
    }

    #[test]
    fn set_triangle_fill_mode_front_back() {
        unrecognized(
            RenderDecoder::set_triangle_fill_mode_front_back,
            &rec(&CmdSetTriangleFillModeFrontBack {
                front_fill_mode: 1,
                back_fill_mode: 0xffff,
            }),
            "setTriangleFrontFillMode:backFillMode:",
        );
    }

    #[test]
    fn set_transform_feedback_state() {
        unrecognized(
            RenderDecoder::set_transform_feedback_state,
            &rec(&CmdSetTransformFeedbackState { state: 2 }),
            "setTransformFeedbackState:",
        );
    }

    /// 0x8d and 0x8e: the index against `maxVertexSamplers`, then the sampler's lookup, then the
    /// raise of a selector the host's encoder does not implement.
    fn sampler_state_case(f: Handler, sel: &'static str) {
        let at = |index, sampler_state_reference| {
            rec(&CmdSetSamplerState {
                sampler_state_reference,
                lod_min_clamp: 0.5,
                lod_max_clamp: 8.0,
                lod_bias: -1.0,
                index,
            })
        };
        unrecognized(f, &at(3, 0), sel);
        // The index first: no lookup.
        {
            let mut r = Rig::new();
            r.set_caps(|c| c.limits.max_vertex_samplers = 3);
            assert_raise(
                f(&mut r.dec, &mut cursor(&at(3, 7))),
                RaiseKind::SamplerIndex,
                "sampler index 3 is past the device's limit",
            );
            assert!(r.calls().is_empty());
            // Then the lookup, then the raise: a failed lookup raises its own first. Without an
            // encoder the lookup still runs.
            let mut r = Rig::new();
            r.provide::<kind::SamplerState>(7);
            let found = f(&mut r.dec, &mut cursor(&at(2, 7))).unwrap_err();
            assert_eq!(found.id(), RaiseId::Platform);
            let failed = f(&mut r.dec, &mut cursor(&at(2, 8))).unwrap_err();
            assert_ne!(failed.id(), RaiseId::Platform);
            assert_eq!(r.sels(), vec!["get_sampler_state", "get_sampler_state"]);
            let mut r = Rig::without_encoder();
            r.provide::<kind::SamplerState>(7);
            f(&mut r.dec, &mut cursor(&at(2, 7))).unwrap();
            assert_eq!(r.sels(), vec!["get_sampler_state"]);
        };
    }

    #[test]
    fn set_vertex_sampler_state() {
        sampler_state_case(
            |d, c| d.set_vertex_sampler_state(Hold::NONE, c),
            "setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
        );
    }

    #[test]
    fn set_fragment_sampler_state() {
        sampler_state_case(
            |d, c| d.set_fragment_sampler_state(Hold::NONE, c),
            "setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
        );
    }

    fn no_read_case(f: Handler, sel: &str) {
        let mut r = Rig::new();
        let mut c = cursor(&[9; 4]);
        f(&mut r.dec, &mut c).unwrap();
        assert_eq!(c.remaining(), 4, "reads nothing");
        assert_eq!(r.encoder_calls(), vec![(sel_static(&r, sel), vec![])]);
        let mut r = Rig::new();
        f(&mut r.dec, &mut cursor(&[])).unwrap();
        let mut r = Rig::without_encoder();
        f(&mut r.dec, &mut cursor(&[])).unwrap();
        assert!(r.calls().is_empty());
    }

    #[test]
    fn set_depth_cleared() {
        no_read_case(RenderDecoder::set_depth_cleared, "setDepthCleared");
    }

    #[test]
    fn set_stencil_cleared() {
        no_read_case(RenderDecoder::set_stencil_cleared, "setStencilCleared");
    }

    #[test]
    fn set_color_resolve_texture() {
        let f: Handler = |d, c| d.set_color_resolve_texture(Hold::NONE, c);
        let at = |i| {
            rec(&CmdSetColorResolveTexture {
                texture_reference: 4,
                slice: 1,
                depth_plane: 2,
                level: 3,
                y_invert: 0x100,
                color_attachment_index: i,
                unknown0e: [0; 2],
            })
        };
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&at(0)[..0xf])));
        // `>=`, so with only attachment 0 (max 0) index 0 fails, before the lookup.
        let mut r = Rig::new();
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(0))),
            RaiseKind::AttachmentIndex,
            "colour attachment index 0 is past the device's limit",
        );
        assert!(r.calls().is_empty());
        let mut r = Rig::new();
        r.dec.max_color_attachment_index = 2;
        assert_raise(
            f(&mut r.dec, &mut cursor(&at(2))),
            RaiseKind::AttachmentIndex,
            "colour attachment index 2 is past the device's limit",
        );
        let t = r.provide::<kind::Texture>(4);
        f(&mut r.dec, &mut cursor(&at(1))).unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:",
                vec![
                    obj(&t),
                    Arg::U64(1),
                    Arg::U64(2),
                    Arg::U64(3),
                    Arg::Bool(true),
                    Arg::U64(1)
                ]
            )]
        );
    }

    #[test]
    fn set_vertex_amplification_mode() {
        simple(
            RenderDecoder::set_vertex_amplification_mode,
            &rec(&CmdSetVertexAmplificationMode { mode: 1, value: 2 }),
            "setVertexAmplificationMode:value:",
            vec![Arg::U64(1), Arg::U64(2)],
        );
    }

    #[test]
    fn set_vertex_amplification_count() {
        let f: Handler = RenderDecoder::set_vertex_amplification_count;
        let mapping = |i: u32| ViewMappingRecord {
            viewport_array_index_offset: i,
            render_target_array_index_offset: i + 1,
        };
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&[0; 3])));
        let mut r = Rig::new();
        r.set_caps(|c| c.max_vertex_amplification_count = 2);
        assert_raise(
            f(
                &mut r.dec,
                &mut cursor(&rec(&CmdSetVertexAmplificationCount { count: 3 })),
            ),
            RaiseKind::VertexAmplificationCount,
            "vertex amplification count 3 is past the device's limit",
        );
        let mut b = rec(&CmdSetVertexAmplificationCount { count: 2 });
        b.extend_from_slice(&rec(&mapping(0)));
        let mut r = Rig::new();
        r.set_caps(|c| c.max_vertex_amplification_count = 2);
        assert_failed_read(f(&mut r.dec, &mut cursor(&b)));
        b.extend_from_slice(&rec(&mapping(5)));
        let mut r = Rig::new();
        r.set_caps(|c| c.max_vertex_amplification_count = 2);
        f(&mut r.dec, &mut cursor(&b)).unwrap();
        let expect = [
            VertexAmplificationViewMapping {
                viewport_array_index_offset: 0,
                render_target_array_index_offset: 1,
            },
            VertexAmplificationViewMapping {
                viewport_array_index_offset: 5,
                render_target_array_index_offset: 6,
            },
        ];
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "setVertexAmplificationCount:viewMappings:",
                vec![Arg::val(&expect[..])]
            )]
        );
    }

    // ------------------------------------------------------------------ tile dimensions
}
