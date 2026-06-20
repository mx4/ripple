//! Ripple — one winit + wgpu window hosting every fluid backend behind the
//! `Simulation` trait, switchable live with the number keys:
//!   1 CPU SPH liquid   2 Eulerian smoke   3 FLIP water   4 GPU SPH
//! An egui panel shows live stats and per-backend tuning. The app owns the
//! shared `Gpu` context, the active `Box<dyn Simulation>`, and the egui overlay.

use std::sync::Arc;
use std::time::Instant;

use ripple::gpu::{
    EguiOverlay, FlipBackend, Gpu, GpuFlipBackend, GpuSmokeBackend, Input, Simulation, SmokeBackend,
    SphBackend, SphCpuBackend,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    sim: Box<dyn Simulation>,
    overlay: EguiOverlay,
    input: Input,
    last: Instant,
    fps: f32,
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ripple — 1 CPU SPH  2 smoke  3 FLIP  4 GPU SPH  5 GPU smoke  6 GPU FLIP")
            .with_inner_size(LogicalSize::new(900.0, 700.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gpu = Gpu::new(window.clone());
        let overlay = EguiOverlay::new(&gpu, &window);
        let sim: Box<dyn Simulation> = Box::new(SphCpuBackend::new(&gpu));
        self.state = Some(State {
            window,
            gpu,
            sim,
            overlay,
            input: Input::default(),
            last: Instant::now(),
            fps: 60.0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else {
            return;
        };
        // Let egui see the event first (so the panel is interactive).
        let _ = st.overlay.on_event(&st.window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                st.gpu.resize(size.width, size.height);
                st.sim.resize(&st.gpu, size.width, size.height);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                let down = state == ElementState::Pressed;
                if down && !repeat {
                    match code {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::Digit1 => st.sim = Box::new(SphCpuBackend::new(&st.gpu)),
                        KeyCode::Digit2 => st.sim = Box::new(SmokeBackend::new(&st.gpu)),
                        KeyCode::Digit3 => st.sim = Box::new(FlipBackend::new(&st.gpu)),
                        KeyCode::Digit4 => st.sim = Box::new(SphBackend::new(&st.gpu)),
                        KeyCode::Digit5 => st.sim = Box::new(GpuSmokeBackend::new(&st.gpu)),
                        KeyCode::Digit6 => st.sim = Box::new(GpuFlipBackend::new(&st.gpu)),
                        _ => {}
                    }
                }
                st.input.key(code, down, repeat);
            }
            WindowEvent::CursorMoved { position, .. } => {
                st.input.mouse = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                st.input.mouse_down = state == ElementState::Pressed;
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - st.last).as_secs_f32();
                st.last = now;
                st.fps = st.fps * 0.9 + (1.0 / dt.max(1e-4)) * 0.1;

                st.sim.update(&st.gpu, dt, &st.input);
                st.input.end_frame();

                if let Some(frame) = st.gpu.acquire() {
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    st.sim.render(&st.gpu, &view);

                    let name = st.sim.name().to_string();
                    let fps = st.fps;
                    let State {
                        gpu,
                        sim,
                        overlay,
                        window,
                        ..
                    } = st;
                    overlay.draw(gpu, window, &view, |ctx| {
                        egui::Window::new("Ripple")
                            .default_pos((10.0, 10.0))
                            .show(ctx, |ui| {
                                ui.label(format!("{name}    {fps:.0} fps"));
                                ui.label("[1] CPU SPH  [2] smoke  [3] FLIP  [4] GPU SPH  [5] GPU smoke  [6] GPU FLIP");
                                ui.separator();
                                sim.ui(gpu, ui);
                            });
                    });
                    frame.present();
                }
                st.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}
