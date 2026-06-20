//! GPU-resident SPH solver. All particle state lives in storage buffers; each
//! solver pass is a compute dispatch. The CPU only updates a small uniform
//! (gravity, dt, …) and records command buffers — no per-frame readback.

use wgpu::util::DeviceExt;

const GRID_CAP: u32 = 64; // max particles tracked per grid cell
const WORKGROUP: u32 = 64;

/// Uniform block shared by every compute pass. Field order/layout must match
/// `struct Params` in `sph.wgsl`. All scalars (4-byte aligned), padded to 96
/// bytes so it satisfies uniform-buffer alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    num: u32,
    cols: u32,
    rows: u32,
    grid_cap: u32,

    h: f32,
    mass: f32,
    poly6: f32,
    spiky_grad: f32,

    visc_lap: f32,
    visc: f32,
    rest_dens: f32,
    stiffness: f32,

    gravity_x: f32,
    gravity_y: f32,
    dt: f32,
    max_speed: f32,

    bound_w: f32,
    bound_h: f32,
    bound_shape: u32,
    bound_damping: f32,

    particle_radius: f32,
    impulse_x: f32,
    impulse_y: f32,
    _pad0: f32,
}

#[inline]
fn workgroups(n: u32) -> u32 {
    n.div_ceil(WORKGROUP)
}

pub struct GpuSim {
    params: Params,
    num: u32,
    cols: u32,
    rows: u32,
    domain: (f32, f32),
    initial_pos: Vec<[f32; 2]>,

    params_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    // Persistent CPU-readback target (tests/debug only).
    readback_buf: wgpu::Buffer,
    // Touched only by the shaders (via the bind group); kept so the GPU buffers
    // stay alive for the sim's lifetime.
    #[allow(dead_code)]
    force_buf: wgpu::Buffer,
    #[allow(dead_code)]
    rho_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pressure_buf: wgpu::Buffer,
    #[allow(dead_code)]
    grid_count_buf: wgpu::Buffer,
    #[allow(dead_code)]
    grid_cells_buf: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
    p_clear: wgpu::ComputePipeline,
    p_build: wgpu::ComputePipeline,
    p_density: wgpu::ComputePipeline,
    p_forces: wgpu::ComputePipeline,
    p_integrate: wgpu::ComputePipeline,
}

impl GpuSim {
    pub fn new(device: &wgpu::Device, w: f32, h: f32) -> Self {
        // Reuse the CPU spawn + rest-density calibration as the single source of
        // truth, so both backends start identically.
        let cpu = crate::sim::Sim::new(w, h);
        let initial_pos = cpu.positions_xy();
        let rest_dens = cpu.rest_density();
        let k = crate::sim::sph_constants();
        let num = initial_pos.len() as u32;

        let cols = (w / k.h).ceil() as u32 + 1;
        let rows = (h / k.h).ceil() as u32 + 1;
        let cells = cols * rows;

        let params = Params {
            num,
            cols,
            rows,
            grid_cap: GRID_CAP,
            h: k.h,
            mass: k.mass,
            poly6: k.poly6,
            spiky_grad: k.spiky_grad,
            visc_lap: k.visc_lap,
            visc: k.visc,
            rest_dens,
            stiffness: k.stiffness,
            gravity_x: 0.0,
            gravity_y: 1200.0,
            dt: 0.0008,
            max_speed: k.max_speed,
            bound_w: w,
            bound_h: h,
            bound_shape: 0,
            bound_damping: k.bound_damping,
            particle_radius: k.particle_radius,
            impulse_x: 0.0,
            impulse_y: 0.0,
            _pad0: 0.0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos"),
            contents: bytemuck::cast_slice(&initial_pos),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let vec2_bytes = (num as u64) * 8;
        let f32_bytes = (num as u64) * 4;
        let vel_buf = mk("vel", vec2_bytes, storage);
        let readback_buf = mk(
            "readback",
            vec2_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let force_buf = mk("force", vec2_bytes, storage);
        let rho_buf = mk("rho", f32_bytes, storage);
        let pressure_buf = mk("pressure", f32_bytes, storage);
        let grid_count_buf = mk("grid_count", (cells as u64) * 4, storage);
        let grid_cells_buf = mk("grid_cells", (cells as u64) * (GRID_CAP as u64) * 4, storage);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sph"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sph.wgsl").into()),
        });

        // One bind-group layout shared by all compute passes.
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sph-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                storage_entry(5),
                storage_entry(6),
                storage_entry(7),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sph-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: force_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rho_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pressure_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grid_cells_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sph-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        GpuSim {
            params,
            num,
            cols,
            rows,
            domain: (w, h),
            initial_pos,
            params_buf,
            pos_buf,
            vel_buf,
            readback_buf,
            force_buf,
            rho_buf,
            pressure_buf,
            grid_count_buf,
            grid_cells_buf,
            bind_group,
            p_clear: make("clear_grid"),
            p_build: make("build_grid"),
            p_density: make("density_pressure"),
            p_forces: make("forces"),
            p_integrate: make("integrate"),
        }
    }

    pub fn num(&self) -> u32 {
        self.num
    }

    pub fn domain(&self) -> (f32, f32) {
        self.domain
    }

    pub fn particle_radius(&self) -> f32 {
        self.params.particle_radius
    }

    pub fn max_speed(&self) -> f32 {
        self.params.max_speed
    }

    pub fn pos_buffer(&self) -> &wgpu::Buffer {
        &self.pos_buf
    }

    pub fn vel_buffer(&self) -> &wgpu::Buffer {
        &self.vel_buf
    }

    /// Set the per-frame dynamic inputs. `impulse` is a one-shot velocity delta
    /// added each substep (pre-divided by the substep count by the caller).
    pub fn set_frame(&mut self, gravity: [f32; 2], dt: f32, impulse: [f32; 2], shape: u32) {
        self.params.gravity_x = gravity[0];
        self.params.gravity_y = gravity[1];
        self.params.dt = dt;
        self.params.impulse_x = impulse[0];
        self.params.impulse_y = impulse[1];
        self.params.bound_shape = shape;
    }

    pub fn upload_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// Number of compute passes per step (used to size timestamp query sets).
    pub const PASSES: usize = 5;

    /// Record one full SPH timestep into `encoder`: clear → build grid →
    /// density → forces → integrate. Each pass is separate so wgpu inserts the
    /// memory barriers each stage depends on. If `qs` is given, begin/end GPU
    /// timestamps are written per pass (slots 0..2*PASSES).
    fn record(&self, encoder: &mut wgpu::CommandEncoder, qs: Option<&wgpu::QuerySet>) {
        let cells = self.cols * self.rows;
        let n = self.num;
        let passes = [
            (&self.p_clear, workgroups(cells)),
            (&self.p_build, workgroups(n)),
            (&self.p_density, workgroups(n)),
            (&self.p_forces, workgroups(n)),
            (&self.p_integrate, workgroups(n)),
        ];
        for (i, (pipe, groups)) in passes.into_iter().enumerate() {
            let timestamp_writes = qs.map(|q| wgpu::ComputePassTimestampWrites {
                query_set: q,
                beginning_of_pass_write_index: Some(i as u32 * 2),
                end_of_pass_write_index: Some(i as u32 * 2 + 1),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    pub fn record_step(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record(encoder, None);
    }

    /// Like [`record_step`](Self::record_step) but writes per-pass GPU
    /// timestamps into `qs` (needs `2 * PASSES` slots). Profiling only —
    /// requires the `TIMESTAMP_QUERY` feature.
    pub fn record_step_timed(&self, encoder: &mut wgpu::CommandEncoder, qs: &wgpu::QuerySet) {
        self.record(encoder, Some(qs));
    }

    /// Restore the initial particle block and zero velocities.
    pub fn reset(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(&self.initial_pos));
        let zeros = vec![0u8; (self.num as usize) * 8];
        queue.write_buffer(&self.vel_buf, 0, &zeros);
    }

    /// Copy positions back to the CPU. For tests/debugging only — the render
    /// path never does this. Uses the persistent `readback_buf`.
    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 2]> {
        let size = (self.num as u64) * 8;
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.pos_buf, 0, &self.readback_buf, 0, size);
        queue.submit([enc.finish()]);

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let out: Vec<[f32; 2]> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.readback_buf.unmap();
        out
    }
}

/// Create a headless wgpu device+queue (no surface). Returns `None` if no GPU
/// adapter is available (so tests can skip rather than fail).
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(wgpu::Features::empty())
}

/// Like [`headless_device`], but also enables whatever subset of `extra` the
/// adapter actually supports (used to opt into `TIMESTAMP_QUERY` for profiling).
pub fn headless_device_with(extra: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("headless"),
        required_features: extra & adapter.features(),
        // Use the adapter's own limits: the grid buffer can be large at big
        // domains, exceeding wgpu's conservative default binding-size limit.
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless check of the GPU solver: run many steps, read positions back,
    /// and assert the fluid is finite, in-bounds, and has settled under gravity
    /// — the same guard that caught the CPU bugs, now for the GPU path.
    #[test]
    fn gpu_solver_is_stable() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping gpu_solver_is_stable");
            return;
        };
        let (w, h) = (900.0, 600.0);
        let mut gpu = GpuSim::new(&device, w, h);
        gpu.set_frame([0.0, 1200.0], 0.0008, [0.0, 0.0], 0);
        gpu.upload_params(&queue);

        // 5000 steps of 0.0008s = 4s simulated. Submit in chunks to keep each
        // command buffer reasonable.
        let mut done = 0;
        while done < 5000 {
            let mut enc = device.create_command_encoder(&Default::default());
            for _ in 0..250 {
                gpu.record_step(&mut enc);
            }
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            done += 250;
        }

        let pos = gpu.read_positions(&device, &queue);
        assert_eq!(pos.len(), gpu.num() as usize);

        let mut avg_y = 0.0;
        for p in &pos {
            assert!(p[0].is_finite() && p[1].is_finite(), "non-finite pos {p:?}");
            assert!(p[0] >= -2.0 && p[0] <= w + 2.0, "escaped x: {p:?}");
            assert!(p[1] >= -2.0 && p[1] <= h + 2.0, "escaped y: {p:?}");
            avg_y += p[1];
        }
        avg_y /= pos.len() as f32;
        assert!(avg_y > h * 0.5, "fluid didn't settle under gravity (avg_y {avg_y})");
        println!("gpu ok: n={} avg_y={:.1}", pos.len(), avg_y);
    }

    /// GPU throughput across particle counts (run with `--ignored --nocapture`).
    /// Shows how per-step dispatch/sync overhead amortises as the count grows —
    /// the GPU only pulls ahead of the CPU at large N. The 1800x1200 row is
    /// directly comparable to the CPU `bench`.
    #[test]
    #[ignore]
    fn gpu_bench() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping gpu_bench");
            return;
        };
        println!("\n        n   steps/s   M particle-steps/s");
        for &(w, h) in &[(900.0, 600.0), (1800.0, 1200.0), (3600.0, 2400.0), (7200.0, 4800.0)] {
            let mut gpu = GpuSim::new(&device, w, h);
            gpu.set_frame([0.0, 1200.0], 0.0008, [0.0, 0.0], 0);
            gpu.upload_params(&queue);

            let run = |steps: usize| {
                let mut done = 0;
                while done < steps {
                    let chunk = 250.min(steps - done);
                    let mut enc = device.create_command_encoder(&Default::default());
                    for _ in 0..chunk {
                        gpu.record_step(&mut enc);
                    }
                    queue.submit([enc.finish()]);
                    let _ = device.poll(wgpu::PollType::wait_indefinitely());
                    done += chunk;
                }
            };

            run(100); // warm up
            let steps = 1000;
            let t0 = std::time::Instant::now();
            run(steps);
            let secs = t0.elapsed().as_secs_f64();
            let n = gpu.num() as f64;
            println!(
                "{:>9} {:>9.0} {:>12.1}",
                gpu.num(),
                steps as f64 / secs,
                n * steps as f64 / secs / 1e6
            );
        }
    }

    /// Per-pass GPU timing via timestamp queries (run with `--ignored
    /// --nocapture`). Shows which of the 5 compute passes actually dominate, so
    /// optimisation effort goes where it matters. Measured at steady state.
    #[test]
    #[ignore]
    fn gpu_profile() {
        let Some((device, queue)) = headless_device_with(wgpu::Features::TIMESTAMP_QUERY) else {
            eprintln!("no GPU adapter; skipping gpu_profile");
            return;
        };
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("TIMESTAMP_QUERY unsupported on this adapter; skipping gpu_profile");
            return;
        }
        let period = queue.get_timestamp_period() as f64; // ns per tick

        let (w, h) = (3600.0, 2400.0); // ~21k particles
        let mut gpu = GpuSim::new(&device, w, h);
        gpu.set_frame([0.0, 1200.0], 0.0008, [0.0, 0.0], 0);
        gpu.upload_params(&queue);

        let slots = (2 * GpuSim::PASSES) as u32;
        let ts_bytes = (slots as u64) * 8;
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: slots,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-resolve"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-staging"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Warm up to a settled, packed state so the neighbour loops are realistic.
        let mut warm = 0;
        while warm < 1000 {
            let mut e = device.create_command_encoder(&Default::default());
            for _ in 0..200 {
                gpu.record_step(&mut e);
            }
            queue.submit([e.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            warm += 200;
        }

        let names = ["clear_grid", "build_grid", "density", "forces", "integrate"];
        let mut sums = [0.0f64; GpuSim::PASSES];
        let runs = 300;
        for _ in 0..runs {
            let mut e = device.create_command_encoder(&Default::default());
            gpu.record_step_timed(&mut e, &qs);
            e.resolve_query_set(&qs, 0..slots, &resolve, 0);
            e.copy_buffer_to_buffer(&resolve, 0, &staging, 0, ts_bytes);
            queue.submit([e.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());

            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let ts: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            for i in 0..GpuSim::PASSES {
                sums[i] += ts[2 * i + 1].wrapping_sub(ts[2 * i]) as f64 * period;
            }
        }

        let avg: Vec<f64> = sums.iter().map(|s| s / runs as f64).collect();
        let total: f64 = avg.iter().sum();
        println!("\nGPU per-pass timing (n={}, avg of {runs} steps):", gpu.num());
        for i in 0..GpuSim::PASSES {
            println!(
                "  {:<10} {:>8.2} us   {:>5.1}%",
                names[i],
                avg[i] / 1000.0,
                100.0 * avg[i] / total
            );
        }
        println!("  {:<10} {:>8.2} us/step", "TOTAL", total / 1000.0);
    }
}
