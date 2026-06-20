//! GPU Eulerian smoke solver (Stam). Mirrors `crate::smoke::Smoke` but runs the
//! whole step in wgpu compute shaders (`smoke.wgsl`), state resident in storage
//! buffers. The pressure projection uses a ping-pong Jacobi relaxation — the
//! reusable grid-solve primitive.

use crate::gpu::{Gpu, Input, Simulation};
use winit::keyboard::KeyCode;

const ITERS: usize = 40; // Jacobi iterations per projection (must be even)
const WG: u32 = 8;
const SMOKE_N: u32 = 256; // GPU grid resolution
const SMOKE_DISPLAY_K: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    dt: f32,
    buoyancy: f32,
    dissipation: f32,
    src_amount: f32,
    src_vy: f32,
    src_half: u32,
    src_y: u32,
}

pub struct GpuSmoke {
    n: u32,
    params: Params,
    params_buf: wgpu::Buffer,
    // grid fields
    u: wgpu::Buffer,
    v: wgpu::Buffer,
    nu: wgpu::Buffer,
    nv: wgpu::Buffer,
    dens: wgpu::Buffer,
    ndens: wgpu::Buffer,
    p: wgpu::Buffer,
    q: wgpu::Buffer,
    div: wgpu::Buffer,
    // bind group with (p@7, q@8) and the swapped one for jacobi ping-pong
    bg_main: wgpu::BindGroup,
    bg_swap: wgpu::BindGroup,
    p_forces: wgpu::ComputePipeline,
    p_div: wgpu::ComputePipeline,
    p_jacobi: wgpu::ComputePipeline,
    p_grad: wgpu::ComputePipeline,
    p_advect_vel: wgpu::ComputePipeline,
    p_advect_dens: wgpu::ComputePipeline,
}

#[inline]
fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

impl GpuSmoke {
    pub fn new(device: &wgpu::Device, n: u32) -> Self {
        let cells = ((n + 2) * (n + 2)) as u64;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cells * 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let u = mk("u");
        let v = mk("v");
        let nu = mk("nu");
        let nv = mk("nv");
        let dens = mk("dens");
        let ndens = mk("ndens");
        let p = mk("p");
        let q = mk("q");
        let div = mk("div");

        let params = Params {
            n,
            dt: 1.0 / 30.0,
            buoyancy: 1.0,
            dissipation: 0.4,
            src_amount: 40.0,
            src_vy: 6.0,
            src_half: (n / 20).max(3),
            src_y: 4,
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for b in 1..=9 {
            entries.push(storage_entry(b));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke-bgl"),
            entries: &entries,
        });

        // p_read at binding 7, q_write at binding 8.
        let make_bg = |p_read: &wgpu::Buffer, q_write: &wgpu::Buffer| {
            let bufs = [
                (1u32, &u),
                (2, &v),
                (3, &nu),
                (4, &nv),
                (5, &dens),
                (6, &ndens),
                (7, p_read),
                (8, q_write),
                (9, &div),
            ];
            let mut e = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }];
            for (b, buf) in bufs {
                e.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: buf.as_entire_binding(),
                });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smoke-bg"),
                layout: &layout,
                entries: &e,
            })
        };
        let bg_main = make_bg(&p, &q);
        let bg_swap = make_bg(&q, &p);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoke"),
            source: wgpu::ShaderSource::Wgsl(include_str!("smoke.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smoke-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // Caller must `upload_params(queue)` once after construction.
        GpuSmoke {
            n,
            params,
            params_buf,
            u,
            v,
            nu,
            nv,
            dens,
            ndens,
            p,
            q,
            div,
            bg_main,
            bg_swap,
            p_forces: make("forces"),
            p_div: make("divergence"),
            p_jacobi: make("jacobi"),
            p_grad: make("subtract_gradient"),
            p_advect_vel: make("advect_vel"),
            p_advect_dens: make("advect_dens"),
        }
    }

    pub fn n(&self) -> u32 {
        self.n
    }

    /// Set the dynamic look knobs (call `upload_params` after).
    pub fn set_params(&mut self, buoyancy: f32, dissipation: f32) {
        self.params.buoyancy = buoyancy;
        self.params.dissipation = dissipation;
    }

    /// Turn the continuous bottom source on/off (call `upload_params` after).
    pub fn set_source(&mut self, on: bool) {
        self.params.src_amount = if on { 40.0 } else { 0.0 };
        self.params.src_vy = if on { 6.0 } else { 0.0 };
    }

    pub fn upload_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn dens_buffer(&self) -> &wgpu::Buffer {
        &self.dens
    }

    pub fn reset(&self, queue: &wgpu::Queue) {
        let zeros = vec![0u8; ((self.n + 2) * (self.n + 2)) as usize * 4];
        for b in [&self.u, &self.v, &self.dens, &self.p, &self.q, &self.div] {
            queue.write_buffer(b, 0, &zeros);
        }
    }

    fn pass(&self, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup) {
        let g = groups(self.n);
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        p.set_pipeline(pipe);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups(g, g, 1);
    }

    fn project(&self, enc: &mut wgpu::CommandEncoder) {
        self.pass(enc, &self.p_div, &self.bg_main);
        for it in 0..ITERS {
            let bg = if it % 2 == 0 { &self.bg_main } else { &self.bg_swap };
            self.pass(enc, &self.p_jacobi, bg);
        }
        // ITERS even -> final pressure ends up in `p` (read by bg_main).
        self.pass(enc, &self.p_grad, &self.bg_main);
    }

    pub fn record_step(&self, enc: &mut wgpu::CommandEncoder) {
        let bytes = ((self.n + 2) * (self.n + 2)) as u64 * 4;
        self.pass(enc, &self.p_forces, &self.bg_main);
        self.project(enc);
        self.pass(enc, &self.p_advect_vel, &self.bg_main);
        enc.copy_buffer_to_buffer(&self.nu, 0, &self.u, 0, bytes);
        enc.copy_buffer_to_buffer(&self.nv, 0, &self.v, 0, bytes);
        self.project(enc);
        self.pass(enc, &self.p_advect_dens, &self.bg_main);
        enc.copy_buffer_to_buffer(&self.ndens, 0, &self.dens, 0, bytes);
    }
}

// ===========================================================================
// Rendering: fullscreen blit of the density buffer
// ===========================================================================
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    n: u32,
    display_k: f32,
    _p0: u32,
    _p1: u32,
}

struct SmokeBufferRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl SmokeBufferRenderer {
    fn new(gpu: &Gpu, dens: &wgpu::Buffer, n: u32) -> Self {
        let params = RenderParams {
            n,
            display_k: SMOKE_DISPLAY_K,
            _p0: 0,
            _p1: 0,
        };
        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke-render-params"),
            size: std::mem::size_of::<RenderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("smoke-render-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smoke-render-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dens.as_entire_binding(),
                },
            ],
        });
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoke-render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("smoke_render.wgsl").into()),
        });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("smoke-render-pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("smoke-render"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self {
            pipeline,
            bind_group,
        }
    }

    fn render(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("smoke-render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ===========================================================================
// Backend
// ===========================================================================
pub struct GpuSmokeBackend {
    smoke: GpuSmoke,
    renderer: SmokeBufferRenderer,
    buoyancy: f32,
    dissipation: f32,
    source_on: bool,
}

impl GpuSmokeBackend {
    pub fn new(gpu: &Gpu) -> Self {
        let smoke = GpuSmoke::new(&gpu.device, SMOKE_N);
        let renderer = SmokeBufferRenderer::new(gpu, smoke.dens_buffer(), SMOKE_N);
        Self {
            smoke,
            renderer,
            buoyancy: 1.0,
            dissipation: 0.4,
            source_on: true,
        }
    }
}

impl Simulation for GpuSmokeBackend {
    fn update(&mut self, gpu: &Gpu, _dt: f32, input: &Input) {
        if input.pressed(KeyCode::KeyS) {
            self.source_on = !self.source_on;
        }
        if input.pressed(KeyCode::KeyR) {
            self.smoke.reset(&gpu.queue);
        }
        self.smoke.set_params(self.buoyancy, self.dissipation);
        self.smoke.set_source(self.source_on);
        self.smoke.upload_params(&gpu.queue);

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("smoke-step") });
        self.smoke.record_step(&mut enc);
        gpu.queue.submit([enc.finish()]);
    }

    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView) {
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("smoke-draw") });
        self.renderer.render(&mut enc, view);
        gpu.queue.submit([enc.finish()]);
    }

    fn name(&self) -> &str {
        "GPU smoke"
    }

    fn ui(&mut self, _gpu: &Gpu, ui: &mut egui::Ui) {
        ui.label(format!("grid: {0}x{0}", SMOKE_N));
        ui.add(egui::Slider::new(&mut self.buoyancy, 0.0..=4.0).text("buoyancy"));
        ui.add(egui::Slider::new(&mut self.dissipation, 0.0..=2.0).text("dissipation"));
        ui.checkbox(&mut self.source_on, "bottom source");
        ui.label("S source   R reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::headless_device;

    #[test]
    fn gpu_smoke_rises_and_is_stable() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let n = 64u32;
        let smoke = GpuSmoke::new(&device, n);
        smoke.upload_params(&queue);

        let mut done = 0;
        while done < 200 {
            let mut enc = device.create_command_encoder(&Default::default());
            for _ in 0..10 {
                smoke.record_step(&mut enc);
            }
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            done += 10;
        }

        // Read density back.
        let cells = ((n + 2) * (n + 2)) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: cells * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(smoke.dens_buffer(), 0, &staging, 0, cells * 4);
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let d: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();

        let mut total = 0.0f32;
        let mut wj = 0.0f32;
        let mut maxd = 0.0f32;
        for j in 1..=n {
            for i in 1..=n {
                let val = d[(i + (n + 2) * j) as usize];
                assert!(val.is_finite(), "density NaN/inf");
                total += val;
                wj += val * j as f32;
                maxd = maxd.max(val);
            }
        }
        assert!(total > 1.0, "all smoke vanished ({total})");
        assert!(maxd < 1e4, "density blew up ({maxd})");
        let com_j = wj / total;
        assert!(com_j > 8.0, "smoke did not rise (com_j {com_j})");
        println!("gpu smoke ok: total={total:.0} max={maxd:.2} com_j={com_j:.1}");
    }
}
