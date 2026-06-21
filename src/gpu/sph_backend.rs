//! GPU SPH as a `Simulation` backend — the first one to slot into the unified
//! shell. Wraps the compute solver (`GpuSim`) and the particle renderer
//! (`Renderer`), and owns the SPH controls (shake, gravity, container shape,
//! render mode).

use crate::gpu::{Gpu, GpuSim, Input, RenderMode, Renderer, Simulation};
use winit::keyboard::KeyCode;

const GRAVITY: f32 = 1200.0; // px/s²
const SHAKE: f32 = 650.0; // px/s velocity jolt
const SIM_DT: f32 = 0.0008;
const SIM_SPEED: f32 = 1.0;
const MAX_STEPS: usize = 40;

pub struct SphBackend {
    sim: GpuSim,
    renderer: Renderer,
    g_scale: f32,
    accumulator: f32,
    pending: [f32; 2], // shake impulse to apply next frame
    shape: u32,
    mode: RenderMode,
}

impl SphBackend {
    pub fn new(gpu: &Gpu) -> Self {
        let (w, h) = (gpu.config.width as f32, gpu.config.height as f32);
        let sim = GpuSim::new(&gpu.device, w, h);
        let renderer = Renderer::new(&gpu.device, gpu.format, &sim, gpu.config.width, gpu.config.height);
        Self {
            sim,
            renderer,
            g_scale: 1.0,
            accumulator: 0.0,
            pending: [0.0, 0.0],
            shape: 0,
            mode: RenderMode::default(),
        }
    }
}

impl Simulation for SphBackend {
    fn update(&mut self, gpu: &Gpu, dt_real: f32, input: &Input) {
        // Shake jolts (one-shot); y is screen-down so "up" is -y.
        if input.pressed(KeyCode::ArrowLeft) {
            self.pending[0] -= SHAKE;
        }
        if input.pressed(KeyCode::ArrowRight) {
            self.pending[0] += SHAKE;
        }
        if input.pressed(KeyCode::Space) {
            self.pending[1] -= SHAKE;
        }
        // Gravity strength (held).
        if input.held(KeyCode::ArrowUp) {
            self.g_scale = (self.g_scale - 1.2 * dt_real).max(0.0);
        }
        if input.held(KeyCode::ArrowDown) {
            self.g_scale = (self.g_scale + 1.2 * dt_real).min(2.5);
        }
        if input.pressed(KeyCode::KeyC) {
            self.shape = 1 - self.shape;
        }
        if input.pressed(KeyCode::KeyM) {
            self.mode = self.mode.next();
        }
        if input.pressed(KeyCode::KeyR) {
            self.g_scale = 1.0;
            self.sim.reset(&gpu.queue);
        }

        // Advance in real time via an accumulator of fixed substeps.
        self.accumulator += dt_real.min(0.05) * SIM_SPEED;
        let mut steps = 0;
        while self.accumulator >= SIM_DT && steps < MAX_STEPS {
            self.accumulator -= SIM_DT;
            steps += 1;
        }
        if steps == MAX_STEPS {
            self.accumulator = 0.0;
        }
        if steps > 0 {
            let gravity = [0.0, GRAVITY * self.g_scale]; // straight down
            let impulse = [self.pending[0] / steps as f32, self.pending[1] / steps as f32];
            self.pending = [0.0, 0.0];
            self.sim.set_frame(gravity, SIM_DT, impulse, self.shape);
            self.sim.upload_params(&gpu.queue);
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sph-step") });
            for _ in 0..steps {
                self.sim.record_step(&mut enc);
            }
            // Reconcile the ping-pong result back into the buffer the renderer
            // binds (once per frame, not per substep).
            self.sim.record_sync(&mut enc);
            gpu.queue.submit([enc.finish()]);
        }
    }

    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView) {
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sph-render") });
        self.renderer.render(&mut enc, view, self.mode);
        gpu.queue.submit([enc.finish()]);
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        // The sim domain is fixed at startup size; the field texture follows the surface.
        self.renderer.resize(&gpu.device, width, height);
    }

    fn name(&self) -> &str {
        "GPU SPH"
    }

    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        ui.label(format!("particles: {}", self.sim.num()));
        ui.add(egui::Slider::new(&mut self.g_scale, 0.0..=2.5).text("gravity"));
        ui.horizontal(|ui| {
            ui.label("render:");
            ui.selectable_value(&mut self.mode, RenderMode::Dots, "dots");
            ui.selectable_value(&mut self.mode, RenderMode::Metaballs, "metaballs");
            ui.selectable_value(&mut self.mode, RenderMode::MarchingSquares, "MS lines");
            ui.selectable_value(&mut self.mode, RenderMode::MarchingSquaresFill, "MS fill");
        });
        ui.horizontal(|ui| {
            if ui.button("toggle shape").clicked() {
                self.shape = 1 - self.shape;
            }
            if ui.button("reset").clicked() {
                self.g_scale = 1.0;
                self.sim.reset(&gpu.queue);
            }
        });
        ui.label("keys: <- -> / Space shake, up/down gravity, C shape, M mode, R reset");
    }
}
