//! GPU FLIP/PIC water (`flip.wgsl`). All state — particles + the staggered MAC
//! grid — lives in storage buffers; the whole step runs in compute. The pressure
//! projection is a red-black Gauss-Seidel SOR solve on a MAC grid with a free
//! surface (it converges far faster than the plain Jacobi sweep used by GPU
//! smoke, so it reaches lower divergence in far fewer passes). See `crate::flip`
//! for the CPU reference.

use crate::gpu::{Gpu, Input, Simulation};
use wgpu::util::DeviceExt;
use winit::keyboard::KeyCode;

// Red-black SOR sweeps per step (each = 2 passes). 10 sweeps converges further
// than the old 40 Jacobi iterations (less volume loss) at ~1.8x the throughput;
// see `gpu_flip_bench`.
const DEFAULT_SWEEPS: usize = 10;
const NX: u32 = 128;
const NY: u32 = 128;
const H: f32 = 1.0;
const GRAVITY: f32 = 9.0;
const SHAKE: f32 = 14.0;
const SIM_DT: f32 = 1.0 / 120.0;
const FLIP_RATIO: f32 = 0.9;
const MAX_SUB: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    num: u32,
    nx: u32,
    ny: u32,
    pad0: u32,
    h: f32,
    gx: f32,
    gy: f32,
    dt: f32,
    flip: f32,
    impulse_x: f32,
    impulse_y: f32,
    pad3: f32,
}

pub struct GpuFlip {
    params: Params,
    nx: u32,
    ny: u32,
    h: f32,
    num: u32,
    sweeps: usize, // red-black SOR sweeps per step (each = 2 passes; dominant cost)
    initial_pos: Vec<[f32; 2]>,

    params_buf: wgpu::Buffer,
    pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    #[allow(dead_code)]
    keep: Vec<wgpu::Buffer>, // u,v,prev,accumulators,s,fluid,div kept alive

    bg_main: wgpu::BindGroup,
    p_integrate: wgpu::ComputePipeline,
    p_clear: wgpu::ComputePipeline,
    p_p2g: wgpu::ComputePipeline,
    p_normalize: wgpu::ComputePipeline,
    p_div: wgpu::ComputePipeline,
    p_sor_red: wgpu::ComputePipeline,
    p_sor_black: wgpu::ComputePipeline,
    p_grad: wgpu::ComputePipeline,
    p_g2p: wgpu::ComputePipeline,
}

#[inline]
fn wg1(n: u32) -> u32 {
    n.div_ceil(64)
}
#[inline]
fn wg2(n: u32) -> u32 {
    n.div_ceil(8)
}

impl GpuFlip {
    pub fn new(device: &wgpu::Device, nx: u32, ny: u32, h: f32) -> Self {
        // Reuse the CPU spawn for identical initial particles.
        let cpu = crate::flip::FlipSim::new(nx as usize, ny as usize, h);
        let initial_pos: Vec<[f32; 2]> = cpu.particles().map(|(p, _)| p).collect();
        let num = initial_pos.len() as u32;

        let usz = ((nx + 1) * ny) as u64;
        let vsz = (nx * (ny + 1)) as u64;
        let csz = (nx * ny) as u64;

        let st = wgpu::BufferUsages::STORAGE;
        let mk = |label: &str, len: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: len * 4,
                usage: st,
                mapped_at_creation: false,
            })
        };
        let pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flip-pos"),
            contents: bytemuck::cast_slice(&initial_pos),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip-vel"),
            size: (num as u64) * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u = mk("u", usz);
        let v = mk("v", vsz);
        let u_prev = mk("u_prev", usz);
        let v_prev = mk("v_prev", vsz);
        let au = mk("au", usz);
        let av = mk("av", vsz);
        let wu = mk("wu", usz);
        let wv = mk("wv", vsz);
        let div = mk("div", csz);
        let p = mk("p", csz);
        let p2 = mk("p2", csz);
        let fluid = mk("fluid", csz);

        // Solid mask: border cells solid (0), interior 1.
        let mut s_mask = vec![1.0f32; (nx * ny) as usize];
        for j in 0..ny {
            for i in 0..nx {
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                    s_mask[(i + nx * j) as usize] = 0.0;
                }
            }
        }
        let s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flip-s"),
            contents: bytemuck::cast_slice(&s_mask),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params = Params {
            num,
            nx,
            ny,
            pad0: 0,
            h,
            gx: 0.0,
            gy: -GRAVITY,
            dt: SIM_DT,
            flip: FLIP_RATIO,
            impulse_x: 0.0,
            impulse_y: 0.0,
            pad3: 0.0,
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
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
        for b in 1..=15 {
            entries.push(storage_entry(b));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flip-bgl"),
            entries: &entries,
        });

        // p at binding 14, p2 at binding 15 (swapped for the ping-pong group).
        let make_bg = |b14: &wgpu::Buffer, b15: &wgpu::Buffer| {
            let bufs = [
                (1u32, &pos),
                (2, &vel),
                (3, &u),
                (4, &v),
                (5, &u_prev),
                (6, &v_prev),
                (7, &au),
                (8, &av),
                (9, &wu),
                (10, &wv),
                (11, &s),
                (12, &fluid),
                (13, &div),
                (14, b14),
                (15, b15),
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
                label: Some("flip-bg"),
                layout: &layout,
                entries: &e,
            })
        };
        // SOR updates `p` in place, so one bind group suffices (no ping-pong).
        let bg_main = make_bg(&p, &p2);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flip"),
            source: wgpu::ShaderSource::Wgsl(include_str!("flip.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flip-pl"),
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

        GpuFlip {
            params,
            nx,
            ny,
            h,
            num,
            sweeps: DEFAULT_SWEEPS,
            initial_pos,
            params_buf,
            pos,
            vel,
            keep: vec![u, v, u_prev, v_prev, au, av, wu, wv, div, p, p2, fluid, s],
            bg_main,
            p_integrate: make("integrate"),
            p_clear: make("clear"),
            p_p2g: make("p2g"),
            p_normalize: make("normalize"),
            p_div: make("divergence"),
            p_sor_red: make("sor_red"),
            p_sor_black: make("sor_black"),
            p_grad: make("subtract_gradient"),
            p_g2p: make("g2p"),
        }
    }

    pub fn num(&self) -> u32 {
        self.num
    }
    pub fn domain(&self) -> (f32, f32) {
        (self.nx as f32 * self.h, self.ny as f32 * self.h)
    }
    pub fn pos_buffer(&self) -> &wgpu::Buffer {
        &self.pos
    }
    pub fn vel_buffer(&self) -> &wgpu::Buffer {
        &self.vel
    }

    pub fn set_frame(&mut self, gravity: [f32; 2], dt: f32, impulse: [f32; 2]) {
        self.params.gx = gravity[0];
        self.params.gy = gravity[1];
        self.params.dt = dt;
        self.params.impulse_x = impulse[0];
        self.params.impulse_y = impulse[1];
    }

    pub fn upload_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn reset(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.pos, 0, bytemuck::cast_slice(&self.initial_pos));
        let zeros = vec![0u8; (self.num as usize) * 8];
        queue.write_buffer(&self.vel, 0, &zeros);
    }

    /// Red-black SOR sweeps per step (each is 2 passes: red then black). This is
    /// the dominant pass count, so it's the main perf/quality knob.
    pub fn set_sweeps(&mut self, n: usize) {
        self.sweeps = n;
    }

    fn pass1d(&self, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, n: u32) {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        p.set_pipeline(pipe);
        p.set_bind_group(0, &self.bg_main, &[]);
        p.dispatch_workgroups(wg1(n), 1, 1);
    }

    fn pass2d(&self, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup) {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        p.set_pipeline(pipe);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups(wg2(self.nx), wg2(self.ny), 1);
    }

    pub fn record_step(&self, enc: &mut wgpu::CommandEncoder) {
        let usz = (self.nx + 1) * self.ny;
        let vsz = self.nx * (self.ny + 1);
        let faces = usz.max(vsz);
        let cells = self.nx * self.ny;

        self.pass1d(enc, &self.p_integrate, self.num);
        self.pass1d(enc, &self.p_clear, faces.max(cells));
        self.pass1d(enc, &self.p_p2g, self.num);
        self.pass1d(enc, &self.p_normalize, faces);
        self.pass2d(enc, &self.p_div, &self.bg_main);
        // Red-black SOR: each sweep updates one colour then the other, in place.
        for _ in 0..self.sweeps {
            self.pass2d(enc, &self.p_sor_red, &self.bg_main);
            self.pass2d(enc, &self.p_sor_black, &self.bg_main);
        }
        self.pass1d(enc, &self.p_grad, faces);
        self.pass1d(enc, &self.p_g2p, self.num);
    }
}

// ===========================================================================
// Particle renderer (reads pos/vel buffers directly; y-up mapping)
// ===========================================================================
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    domain_w: f32,
    domain_h: f32,
    radius: f32,
    max_speed: f32,
}

struct FlipRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl FlipRenderer {
    fn new(gpu: &Gpu, flip: &GpuFlip) -> Self {
        let (w, h) = flip.domain();
        let params = RenderParams {
            domain_w: w,
            domain_h: h,
            radius: H * 0.45,
            max_speed: 25.0,
        };
        let params_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flip-render-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let read = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flip-render-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    read(1),
                    read(2),
                ],
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flip-render-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flip.pos_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flip.vel_buffer().as_entire_binding(),
                },
            ],
        });
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flip-render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("flip_render.wgsl").into()),
        });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("flip-render-pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("flip-render"),
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

    fn render(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, count: u32) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flip-render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.06,
                        b: 0.09,
                        a: 1.0,
                    }),
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
        pass.draw(0..6, 0..count);
    }
}

// ===========================================================================
// Backend
// ===========================================================================
pub struct GpuFlipBackend {
    flip: GpuFlip,
    renderer: FlipRenderer,
    g_scale: f32,
    accum: f32,
    pending: [f32; 2],
}

impl GpuFlipBackend {
    pub fn new(gpu: &Gpu) -> Self {
        let flip = GpuFlip::new(&gpu.device, NX, NY, H);
        let renderer = FlipRenderer::new(gpu, &flip);
        Self {
            flip,
            renderer,
            g_scale: 1.0,
            accum: 0.0,
            pending: [0.0, 0.0],
        }
    }
}

impl Simulation for GpuFlipBackend {
    fn update(&mut self, gpu: &Gpu, dt_real: f32, input: &Input) {
        if input.pressed(KeyCode::ArrowLeft) {
            self.pending[0] -= SHAKE;
        }
        if input.pressed(KeyCode::ArrowRight) {
            self.pending[0] += SHAKE;
        }
        if input.pressed(KeyCode::Space) {
            self.pending[1] += SHAKE; // y up
        }
        if input.held(KeyCode::ArrowUp) {
            self.g_scale = (self.g_scale - 1.2 * dt_real).max(0.0);
        }
        if input.held(KeyCode::ArrowDown) {
            self.g_scale = (self.g_scale + 1.2 * dt_real).min(2.5);
        }
        if input.pressed(KeyCode::KeyR) {
            self.g_scale = 1.0;
            self.flip.reset(&gpu.queue);
        }

        self.accum += dt_real.min(0.05);
        let mut steps = 0;
        while self.accum >= SIM_DT && steps < MAX_SUB {
            self.accum -= SIM_DT;
            steps += 1;
        }
        if steps == MAX_SUB {
            self.accum = 0.0;
        }
        if steps > 0 {
            let impulse = [self.pending[0] / steps as f32, self.pending[1] / steps as f32];
            self.pending = [0.0, 0.0];
            self.flip
                .set_frame([0.0, -GRAVITY * self.g_scale], SIM_DT, impulse);
            self.flip.upload_params(&gpu.queue);
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("flip-step") });
            for _ in 0..steps {
                self.flip.record_step(&mut enc);
            }
            gpu.queue.submit([enc.finish()]);
        }
    }

    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView) {
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("flip-draw") });
        self.renderer.render(&mut enc, view, self.flip.num());
        gpu.queue.submit([enc.finish()]);
    }

    fn name(&self) -> &str {
        "GPU FLIP"
    }

    fn ui(&mut self, _gpu: &Gpu, ui: &mut egui::Ui) {
        ui.label(format!("particles: {}   grid: {NX}x{NY}", self.flip.num()));
        ui.add(egui::Slider::new(&mut self.g_scale, 0.0..=2.5).text("gravity"));
        ui.label("keys: <- -> / Space shake, up/down gravity, R reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::headless_device;

    #[test]
    fn gpu_flip_settles_and_is_stable() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (nx, ny, h) = (64u32, 64u32, 1.0);
        let mut flip = GpuFlip::new(&device, nx, ny, h);
        flip.set_frame([0.0, -GRAVITY], SIM_DT, [0.0, 0.0]);
        flip.upload_params(&queue);
        let (w, hgt) = flip.domain();
        let num = flip.num();
        assert!(num > 500, "expected a decent block of particles");

        let mut done = 0;
        while done < 600 {
            let mut enc = device.create_command_encoder(&Default::default());
            for _ in 0..30 {
                flip.record_step(&mut enc);
            }
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            done += 30;
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (num as u64) * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(flip.pos_buffer(), 0, &staging, 0, (num as u64) * 8);
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pos: Vec<[f32; 2]> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();

        let mut avg_y = 0.0f32;
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        for p in &pos {
            assert!(p[0].is_finite() && p[1].is_finite(), "particle NaN/inf");
            assert!(p[0] > 0.0 && p[0] < w, "escaped x: {}", p[0]);
            assert!(p[1] > 0.0 && p[1] < hgt, "escaped y: {}", p[1]);
            avg_y += p[1];
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
        }
        avg_y /= num as f32;
        assert!(avg_y < hgt * 0.45, "water did not settle (avg_y {avg_y})");
        assert!(max_x - min_x > w * 0.3, "water didn't spread ({})", max_x - min_x);
        println!("gpu flip ok: n={num} avg_y={avg_y:.1} width={:.1}", max_x - min_x);
    }

    /// Sweep the Jacobi iteration count (the dominant pass count) at the app's
    /// 128x128 grid: throughput vs. solution quality. The settled water should
    /// stay low (avg_y) and spread wide (width) — if those hold at fewer
    /// iterations, the extra passes are wasted dispatch overhead. Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_flip_bench() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping gpu_flip_bench");
            return;
        };
        let (nx, ny, h) = (128u32, 128u32, 1.0);
        let w = nx as f32 * h;

        // Reference: old 40-iter Jacobi settled to avg_y ~= 6.0 at 599 steps/s.
        // Quality target is matching that avg_y (more = less volume loss).
        println!("\n  sweeps   passes/step   steps/s   avg_y(ref~6.0)   width(spread>{:.0})", w * 0.3);
        for &sweeps in &[20usize, 14, 10, 8, 6, 4] {
            let mut flip = GpuFlip::new(&device, nx, ny, h);
            flip.set_sweeps(sweeps);
            flip.set_frame([0.0, -GRAVITY], SIM_DT, [0.0, 0.0]);
            flip.upload_params(&queue);
            let num = flip.num();

            // Settle the water (timed quality is measured at steady state).
            let mut done = 0;
            while done < 600 {
                let mut enc = device.create_command_encoder(&Default::default());
                for _ in 0..30 {
                    flip.record_step(&mut enc);
                }
                queue.submit([enc.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                done += 30;
            }

            // Time throughput.
            let steps = 600;
            let t0 = std::time::Instant::now();
            let mut d = 0;
            while d < steps {
                let mut enc = device.create_command_encoder(&Default::default());
                for _ in 0..30 {
                    flip.record_step(&mut enc);
                }
                queue.submit([enc.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                d += 30;
            }
            let sps = steps as f64 / t0.elapsed().as_secs_f64();

            // Read back quality.
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: (num as u64) * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(flip.pos_buffer(), 0, &staging, 0, (num as u64) * 8);
            queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let pos: Vec<[f32; 2]> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            let mut avg_y = 0.0f32;
            let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
            let mut finite = true;
            for p in &pos {
                finite &= p[0].is_finite() && p[1].is_finite();
                avg_y += p[1];
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
            }
            avg_y /= num as f32;
            let passes = 7 + 2 * sweeps;
            println!(
                "  {:>6}   {:>11}   {:>7.0}   {:>10.1}   {:>10.1}{}",
                sweeps, passes, sps, avg_y, max_x - min_x,
                if finite { "" } else { "  NONFINITE!" }
            );
        }
    }
}
