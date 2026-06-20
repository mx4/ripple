//! The CPU solvers (SPH, smoke, FLIP) wrapped as `Simulation` backends so they
//! live in the same wgpu app as the GPU SPH. Each steps its pure-math solver on
//! the CPU, then uploads results to a GPU renderer (particles or a field texture).

use crate::flip::FlipSim;
use crate::gpu::{FieldRenderer, Gpu, Input, ParticleRenderer, Simulation};
use crate::sim::{Bounds, Shape, Sim, PARTICLE_RADIUS};
use crate::smoke::Smoke;
use glam::vec2;
use winit::keyboard::KeyCode;

const MAX_STEPS: usize = 40;

// ===========================================================================
// CPU SPH liquid
// ===========================================================================
const SPH_GRAVITY: f32 = 1200.0;
const SPH_SHAKE: f32 = 650.0;
const SPH_DT: f32 = 0.0008;
const SPH_SPEED: f32 = 1.0;

pub struct SphCpuBackend {
    sim: Sim,
    bounds: Bounds,
    g_scale: f32,
    accum: f32,
    renderer: ParticleRenderer,
    pos: Vec<[f32; 2]>,
    vel: Vec<[f32; 2]>,
}

impl SphCpuBackend {
    pub fn new(gpu: &Gpu) -> Self {
        let (w, h) = (gpu.config.width as f32, gpu.config.height as f32);
        let sim = Sim::new(w, h);
        let renderer = ParticleRenderer::new(gpu, sim.len() as u32, (w, h), PARTICLE_RADIUS * 0.85, 450.0);
        Self {
            sim,
            bounds: Bounds { w, h, shape: Shape::Rect },
            g_scale: 1.0,
            accum: 0.0,
            renderer,
            pos: Vec::new(),
            vel: Vec::new(),
        }
    }
}

impl Simulation for SphCpuBackend {
    fn update(&mut self, _gpu: &Gpu, dt_real: f32, input: &Input) {
        if input.pressed(KeyCode::ArrowLeft) {
            self.sim.add_impulse(vec2(-SPH_SHAKE, 0.0));
        }
        if input.pressed(KeyCode::ArrowRight) {
            self.sim.add_impulse(vec2(SPH_SHAKE, 0.0));
        }
        if input.pressed(KeyCode::Space) {
            self.sim.add_impulse(vec2(0.0, -SPH_SHAKE)); // y down: up = -y
        }
        if input.held(KeyCode::ArrowUp) {
            self.g_scale = (self.g_scale - 1.2 * dt_real).max(0.0);
        }
        if input.held(KeyCode::ArrowDown) {
            self.g_scale = (self.g_scale + 1.2 * dt_real).min(2.5);
        }
        if input.pressed(KeyCode::KeyC) {
            self.bounds.shape = match self.bounds.shape {
                Shape::Rect => Shape::Circle,
                Shape::Circle => Shape::Rect,
            };
        }
        if input.pressed(KeyCode::KeyR) {
            self.g_scale = 1.0;
            self.sim.reset(self.bounds.w, self.bounds.h);
        }

        let gravity = vec2(0.0, SPH_GRAVITY * self.g_scale);
        self.accum += dt_real.min(0.05) * SPH_SPEED;
        let mut steps = 0;
        while self.accum >= SPH_DT && steps < MAX_STEPS {
            self.sim.step(SPH_DT, gravity, &self.bounds);
            self.accum -= SPH_DT;
            steps += 1;
        }
        if steps == MAX_STEPS {
            self.accum = 0.0;
        }

        self.pos.clear();
        self.vel.clear();
        self.pos.extend(self.sim.pos.iter().map(|p| [p.x, p.y]));
        self.vel.extend(self.sim.vel.iter().map(|v| [v.x, v.y]));
    }

    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView) {
        let n = self.renderer.upload(gpu, &self.pos, &self.vel);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.renderer.render(&mut enc, view, n);
        gpu.queue.submit([enc.finish()]);
    }

    fn name(&self) -> &str {
        "CPU SPH"
    }

    fn ui(&mut self, _gpu: &Gpu, ui: &mut egui::Ui) {
        ui.label(format!("particles: {}", self.sim.len()));
        ui.add(egui::Slider::new(&mut self.g_scale, 0.0..=2.5).text("gravity"));
        if ui.button("toggle shape").clicked() {
            self.bounds.shape = match self.bounds.shape {
                Shape::Rect => Shape::Circle,
                Shape::Circle => Shape::Rect,
            };
        }
        ui.label("keys: <- -> / Space shake, up/down gravity, C shape");
    }
}

// ===========================================================================
// Eulerian smoke
// ===========================================================================
const SMOKE_N: usize = 128;
const SMOKE_DISPLAY_K: f32 = 1.5;

pub struct SmokeBackend {
    smoke: Smoke,
    renderer: FieldRenderer,
    rgba: Vec<u8>,
    source_on: bool,
}

impl SmokeBackend {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            smoke: Smoke::new(SMOKE_N),
            renderer: FieldRenderer::new(gpu, SMOKE_N as u32),
            rgba: vec![0u8; SMOKE_N * SMOKE_N * 4],
            source_on: true,
        }
    }

    fn cell(px: f32, py: f32, gpu: &Gpu) -> (i32, i32) {
        let i = (px / gpu.config.width as f32 * SMOKE_N as f32) as i32 + 1;
        let j = SMOKE_N as i32 - (py / gpu.config.height as f32 * SMOKE_N as f32) as i32;
        (i.clamp(1, SMOKE_N as i32), j.clamp(1, SMOKE_N as i32))
    }
}

impl Simulation for SmokeBackend {
    fn update(&mut self, gpu: &Gpu, dt_real: f32, input: &Input) {
        let dt = dt_real.min(1.0 / 30.0);
        if input.pressed(KeyCode::KeyS) {
            self.source_on = !self.source_on;
        }
        if input.pressed(KeyCode::KeyV) {
            self.smoke.vorticity = if self.smoke.vorticity > 0.0 { 0.0 } else { 3.0 };
        }
        if input.pressed(KeyCode::KeyR) {
            self.smoke.reset();
        }

        if self.source_on {
            let cx = SMOKE_N / 2;
            for dj in 0..3 {
                for di in -3i32..=3 {
                    let i = (cx as i32 + di).clamp(1, SMOKE_N as i32) as usize;
                    self.smoke.add_density(i, 4 + dj, 40.0);
                    self.smoke.add_velocity(i, 4 + dj, 0.0, 6.0);
                }
            }
        }

        if input.mouse_down {
            let (ci, cj) = Self::cell(input.mouse.0, input.mouse.1, gpu);
            let (dx, dy) = input.mouse_delta();
            let dvx = dx / gpu.config.width as f32 * SMOKE_N as f32 * 3.0;
            let dvy = -dy / gpu.config.height as f32 * SMOKE_N as f32 * 3.0;
            for dj in -2i32..=2 {
                for di in -2i32..=2 {
                    let i = (ci + di).clamp(1, SMOKE_N as i32) as usize;
                    let j = (cj + dj).clamp(1, SMOKE_N as i32) as usize;
                    self.smoke.add_density(i, j, 60.0);
                    self.smoke.add_velocity(i, j, dvx, dvy);
                }
            }
        }

        self.smoke.step(dt);

        for j in 1..=SMOKE_N {
            for i in 1..=SMOKE_N {
                let d = self.smoke.density_at(i, j).max(0.0);
                let b = ((1.0 - (-d * SMOKE_DISPLAY_K).exp()) * 255.0) as u8;
                let x = i - 1;
                let y = SMOKE_N - j; // flip so up is up
                let k = (y * SMOKE_N + x) * 4;
                self.rgba[k] = (b as f32 * 0.9) as u8;
                self.rgba[k + 1] = (b as f32 * 0.95) as u8;
                self.rgba[k + 2] = b;
                self.rgba[k + 3] = 255;
            }
        }
    }

    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView) {
        self.renderer.upload(gpu, &self.rgba);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.renderer.render(&mut enc, view);
        gpu.queue.submit([enc.finish()]);
    }

    fn name(&self) -> &str {
        "smoke"
    }

    fn ui(&mut self, _gpu: &Gpu, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.source_on, "bottom source");
        let mut vort = self.smoke.vorticity > 0.0;
        if ui.checkbox(&mut vort, "vorticity").changed() {
            self.smoke.vorticity = if vort { 3.0 } else { 0.0 };
        }
        ui.label("drag to inject smoke");
    }
}

// ===========================================================================
// FLIP water
// ===========================================================================
const FLIP_NX: usize = 80;
const FLIP_NY: usize = 80;
const FLIP_H: f32 = 1.0;
const FLIP_GRAVITY: f32 = 9.0;
const FLIP_SHAKE: f32 = 14.0;
const FLIP_DT: f32 = 1.0 / 120.0;

pub struct FlipBackend {
    sim: FlipSim,
    g_scale: f32,
    accum: f32,
    renderer: ParticleRenderer,
    pos: Vec<[f32; 2]>,
    vel: Vec<[f32; 2]>,
}

impl FlipBackend {
    pub fn new(gpu: &Gpu) -> Self {
        let sim = FlipSim::new(FLIP_NX, FLIP_NY, FLIP_H);
        let domain = sim.domain();
        let renderer = ParticleRenderer::new(gpu, sim.len() as u32, domain, FLIP_H * 0.45, 25.0);
        Self {
            sim,
            g_scale: 1.0,
            accum: 0.0,
            renderer,
            pos: Vec::new(),
            vel: Vec::new(),
        }
    }
}

impl Simulation for FlipBackend {
    fn update(&mut self, gpu: &Gpu, dt_real: f32, input: &Input) {
        if input.pressed(KeyCode::ArrowLeft) {
            self.sim.add_impulse(-FLIP_SHAKE, 0.0);
        }
        if input.pressed(KeyCode::ArrowRight) {
            self.sim.add_impulse(FLIP_SHAKE, 0.0);
        }
        if input.pressed(KeyCode::Space) {
            self.sim.add_impulse(0.0, FLIP_SHAKE); // y up
        }
        if input.held(KeyCode::ArrowUp) {
            self.g_scale = (self.g_scale - 1.2 * dt_real).max(0.0);
        }
        if input.held(KeyCode::ArrowDown) {
            self.g_scale = (self.g_scale + 1.2 * dt_real).min(2.5);
        }
        if input.pressed(KeyCode::KeyR) {
            self.g_scale = 1.0;
            self.sim.reset();
        }
        self.sim.config().gravity = [0.0, -FLIP_GRAVITY * self.g_scale];

        let (w, h) = self.sim.domain();
        if input.mouse_down {
            let sw = gpu.config.width as f32;
            let sh = gpu.config.height as f32;
            let wx = input.mouse.0 / sw * w;
            let wy = (1.0 - input.mouse.1 / sh) * h; // screen y-down -> world y-up
            let (dx, dy) = input.mouse_delta();
            let dvx = dx / sw * w / FLIP_DT * 0.02;
            let dvy = -dy / sh * h / FLIP_DT * 0.02;
            self.sim.push(wx, wy, dvx, dvy, h * 0.12);
        }

        self.accum += dt_real.min(0.05);
        let mut steps = 0;
        while self.accum >= FLIP_DT && steps < MAX_STEPS {
            self.sim.step(FLIP_DT);
            self.accum -= FLIP_DT;
            steps += 1;
        }
        if steps == MAX_STEPS {
            self.accum = 0.0;
        }

        // Upload with y flipped to domain so the (y-up) sim maps right on screen.
        self.pos.clear();
        self.vel.clear();
        for (p, v) in self.sim.particles() {
            self.pos.push([p[0], h - p[1]]);
            self.vel.push([v[0], v[1]]);
        }
    }

    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView) {
        let n = self.renderer.upload(gpu, &self.pos, &self.vel);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.renderer.render(&mut enc, view, n);
        gpu.queue.submit([enc.finish()]);
    }

    fn name(&self) -> &str {
        "FLIP water"
    }

    fn ui(&mut self, _gpu: &Gpu, ui: &mut egui::Ui) {
        ui.label(format!("particles: {}", self.sim.len()));
        ui.add(egui::Slider::new(&mut self.g_scale, 0.0..=2.5).text("gravity"));
        ui.label("keys: <- -> / Space shake, up/down gravity");
    }
}
