//! Particle renderer with three modes, all reading the simulation's storage
//! buffers directly (no CPU readback):
//!
//! - dots — instanced discs straight to the surface.
//! - metaballs — accumulate soft blobs into an offscreen float field, then a
//!   fullscreen pass thresholds it into a smooth liquid surface.
//! - marching squares — same field, but extract its contour as line segments.
//!
//! Metaballs and marching squares share the blob (field accumulation) pass.

use crate::gpu::GpuSim;
use wgpu::util::DeviceExt;

/// Which way to draw the fluid. Cycled by the app's `M` key.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderMode {
    Dots,
    #[default]
    Metaballs,
    MarchingSquares,
    MarchingSquaresFill,
}

impl RenderMode {
    pub fn next(self) -> Self {
        match self {
            RenderMode::Dots => RenderMode::Metaballs,
            RenderMode::Metaballs => RenderMode::MarchingSquares,
            RenderMode::MarchingSquares => RenderMode::MarchingSquaresFill,
            RenderMode::MarchingSquaresFill => RenderMode::Dots,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            RenderMode::Dots => "dots",
            RenderMode::Metaballs => "metaballs",
            RenderMode::MarchingSquares => "marching squares (lines)",
            RenderMode::MarchingSquaresFill => "marching squares (fill)",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    domain_w: f32,
    domain_h: f32,
    radius: f32,
    max_speed: f32,
}

/// Float field for the metaball accumulation (renderable, blendable, samplable).
const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Marching-squares grid cell size in texels (must match `STEP` in the shader).
const MC_STEP: u32 = 8;
const BG: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.06,
    b: 0.09,
    a: 1.0,
};

pub struct Renderer {
    num: u32,
    // Shared by the dot and blob passes (uniform + pos + vel).
    particle_bind_group: wgpu::BindGroup,
    dot_pipeline: wgpu::RenderPipeline,
    blob_pipeline: wgpu::RenderPipeline,
    // Metaball composite (field -> surface).
    composite_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    // Marching-squares contour + fill (field -> surface, both read field in VS).
    contour_pipeline: wgpu::RenderPipeline,
    fill_pipeline: wgpu::RenderPipeline,
    contour_layout: wgpu::BindGroupLayout,
    contour_bind_group: wgpu::BindGroup,
    // Offscreen density field + its current size.
    field_view: wgpu::TextureView,
    field_w: u32,
    field_h: u32,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sim: &GpuSim,
        width: u32,
        height: u32,
    ) -> Self {
        let (w, h) = sim.domain();
        let params = RenderParams {
            domain_w: w,
            domain_h: h,
            radius: sim.particle_radius(),
            max_speed: sim.max_speed(),
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- particle bind group (uniform + pos + vel), shared by dot & blob ---
        let read_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let particle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                read_storage(1),
                read_storage(2),
            ],
        });
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-bg"),
            layout: &particle_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim.pos_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim.vel_buffer().as_entire_binding(),
                },
            ],
        });
        let particle_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render-pl"),
            bind_group_layouts: &[Some(&particle_layout)],
            immediate_size: 0,
        });

        // --- dot pipeline (-> surface) ---
        let dot_shader = shader(device, "render", include_str!("render.wgsl"));
        let dot_pipeline = pipeline(
            device,
            "dots",
            &particle_pl,
            &dot_shader,
            "vs",
            "fs",
            format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::TriangleList,
        );

        // --- blob pipeline (-> field, additive) ---
        let blob_shader = shader(device, "metaball-blob", include_str!("metaball_blob.wgsl"));
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let blob_pipeline = pipeline(
            device,
            "blobs",
            &particle_pl,
            &blob_shader,
            "vs_blob",
            "fs_blob",
            FIELD_FORMAT,
            Some(wgpu::BlendState {
                color: additive,
                alpha: additive,
            }),
            wgpu::PrimitiveTopology::TriangleList,
        );

        // --- metaball composite (field tex + sampler, FRAGMENT) ---
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite-bgl"),
            entries: &[
                tex_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let composite_shader = shader(
            device,
            "metaball-composite",
            include_str!("metaball_composite.wgsl"),
        );
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite-pl"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });
        let composite_pipeline = pipeline(
            device,
            "composite",
            &composite_pl,
            &composite_shader,
            "vs_full",
            "fs_threshold",
            format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::TriangleList,
        );

        // --- marching-squares contour (field tex in VERTEX, line list) ---
        let contour_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contour-bgl"),
            entries: &[tex_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let contour_shader = shader(
            device,
            "marching-squares",
            include_str!("marching_squares.wgsl"),
        );
        let contour_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("contour-pl"),
            bind_group_layouts: &[Some(&contour_layout)],
            immediate_size: 0,
        });
        let contour_pipeline = pipeline(
            device,
            "contour",
            &contour_pl,
            &contour_shader,
            "vs_contour",
            "fs_contour",
            format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::LineList,
        );

        // --- marching-squares fill (same field bind group, triangles) ---
        let fill_shader = shader(
            device,
            "marching-squares-fill",
            include_str!("marching_squares_fill.wgsl"),
        );
        let fill_pipeline = pipeline(
            device,
            "ms-fill",
            &contour_pl,
            &fill_shader,
            "vs_fill",
            "fs_fill",
            format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::TriangleList,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("field-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let field_view = make_field_view(device, width, height);
        let composite_bind_group = composite_bg(device, &composite_layout, &sampler, &field_view);
        let contour_bind_group = contour_bg(device, &contour_layout, &field_view);

        Renderer {
            num: sim.num(),
            particle_bind_group,
            dot_pipeline,
            blob_pipeline,
            composite_pipeline,
            composite_layout,
            composite_bind_group,
            sampler,
            contour_pipeline,
            fill_pipeline,
            contour_layout,
            contour_bind_group,
            field_view,
            field_w: width.max(1),
            field_h: height.max(1),
        }
    }

    /// Recreate the field texture (and its bind groups) for a new surface size.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.field_view = make_field_view(device, width, height);
        self.composite_bind_group =
            composite_bg(device, &self.composite_layout, &self.sampler, &self.field_view);
        self.contour_bind_group = contour_bg(device, &self.contour_layout, &self.field_view);
        self.field_w = width.max(1);
        self.field_h = height.max(1);
    }

    /// Draw the fluid in the given mode.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        mode: RenderMode,
    ) {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(BG),
                store: wgpu::StoreOp::Store,
            },
        })];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("fluid"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        match mode {
            RenderMode::Dots => {
                let mut pass = encoder.begin_render_pass(&desc);
                pass.set_pipeline(&self.dot_pipeline);
                pass.set_bind_group(0, &self.particle_bind_group, &[]);
                pass.draw(0..6, 0..self.num);
            }
            RenderMode::Metaballs => {
                self.record_blobs(encoder);
                let mut pass = encoder.begin_render_pass(&desc);
                pass.set_pipeline(&self.composite_pipeline);
                pass.set_bind_group(0, &self.composite_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            RenderMode::MarchingSquares => {
                self.record_blobs(encoder);
                // 4 vertices per grid cell (2 segments x 2 endpoints).
                let cells = (self.field_w / MC_STEP) * (self.field_h / MC_STEP);
                let mut pass = encoder.begin_render_pass(&desc);
                pass.set_pipeline(&self.contour_pipeline);
                pass.set_bind_group(0, &self.contour_bind_group, &[]);
                pass.draw(0..cells * 4, 0..1);
            }
            RenderMode::MarchingSquaresFill => {
                self.record_blobs(encoder);
                // 9 vertices per grid cell (up to 3 triangles).
                let cells = (self.field_w / MC_STEP) * (self.field_h / MC_STEP);
                let mut pass = encoder.begin_render_pass(&desc);
                pass.set_pipeline(&self.fill_pipeline);
                pass.set_bind_group(0, &self.contour_bind_group, &[]);
                pass.draw(0..cells * 9, 0..1);
            }
        }
    }

    /// Pass 1 for the field-based modes: accumulate blobs into the field.
    fn record_blobs(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blobs"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.field_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.blob_pipeline);
        pass.set_bind_group(0, &self.particle_bind_group, &[]);
        pass.draw(0..6, 0..self.num);
    }
}

fn tex_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

#[allow(clippy::too_many_arguments)]
fn pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    topology: wgpu::PrimitiveTopology,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn make_field_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("metaball-field"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FIELD_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn composite_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn contour_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("contour-bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(view),
        }],
    })
}
