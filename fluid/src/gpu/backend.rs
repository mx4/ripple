//! The backend abstraction: every fluid model implements `Simulation`, so the
//! app can hold a `Box<dyn Simulation>` and switch between them. Each backend
//! owns its own time-stepping (different solvers want different timesteps), its
//! compute work, and its rendering — the app just feeds it input and a surface.

use crate::gpu::Gpu;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// Per-frame input snapshot the app builds from window events.
#[derive(Default)]
pub struct Input {
    held: HashSet<KeyCode>,
    pressed: HashSet<KeyCode>, // went down this frame (auto-repeat excluded)
    pub mouse: (f32, f32),     // physical pixels
    prev_mouse: (f32, f32),
    pub mouse_down: bool,
}

impl Input {
    pub fn key(&mut self, code: KeyCode, down: bool, repeat: bool) {
        if down {
            if !repeat {
                self.pressed.insert(code);
            }
            self.held.insert(code);
        } else {
            self.held.remove(&code);
        }
    }

    pub fn held(&self, code: KeyCode) -> bool {
        self.held.contains(&code)
    }

    pub fn pressed(&self, code: KeyCode) -> bool {
        self.pressed.contains(&code)
    }

    pub fn mouse_delta(&self) -> (f32, f32) {
        (self.mouse.0 - self.prev_mouse.0, self.mouse.1 - self.prev_mouse.1)
    }

    /// Call after each frame's `update`: clear one-shot state.
    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.prev_mouse = self.mouse;
    }
}

/// A selectable fluid model. Backends advance themselves, then draw to a view.
pub trait Simulation {
    /// Advance the simulation by `dt` wall-clock seconds, consuming `input`.
    fn update(&mut self, gpu: &Gpu, dt: f32, input: &Input);
    /// Draw the current state to `view`.
    fn render(&mut self, gpu: &Gpu, view: &wgpu::TextureView);
    /// React to a surface resize (e.g. recreate size-dependent textures).
    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {}
    /// Short display name for the HUD / window title.
    fn name(&self) -> &str;
    /// Add backend-specific widgets to the egui panel (params, stats, buttons).
    fn ui(&mut self, _gpu: &Gpu, _ui: &mut egui::Ui) {}
}
