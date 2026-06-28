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

/// Build a backend from its number-key digit (1–6), matching the live switcher.
fn make_backend(digit: u8, gpu: &Gpu) -> Box<dyn Simulation> {
    match digit {
        2 => Box::new(SmokeBackend::new(gpu)),
        3 => Box::new(FlipBackend::new(gpu)),
        4 => Box::new(SphBackend::new(gpu)),
        5 => Box::new(GpuSmokeBackend::new(gpu)),
        6 => Box::new(GpuFlipBackend::new(gpu)),
        _ => Box::new(SphCpuBackend::new(gpu)),
    }
}

/// Autonomous screenshot run (set `RIPPLE_CAPTURE=1`): for each entry, show the
/// backend for `warmup` seconds of real-time simulation so the fluid develops,
/// save a clean PNG into `assets/`, then advance. Quits when the plan is done.
/// `(digit, warmup_seconds)` — the three GPU showcase backends, each caught at a
/// flattering moment (the same shots used on the README).
const CAPTURE_PLAN: &[(u8, f32)] = &[(4, 1.6), (6, 3.5), (5, 2.0)];

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    sim: Box<dyn Simulation>,
    overlay: EguiOverlay,
    input: Input,
    last: Instant,
    fps: f32,
    screenshot: bool,
    capture: bool,
    shot_idx: usize,
    phase_start: Instant,
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
        let capture = std::env::var_os("RIPPLE_CAPTURE").is_some();
        let sim: Box<dyn Simulation> = if capture {
            make_backend(CAPTURE_PLAN[0].0, &gpu)
        } else {
            Box::new(SphCpuBackend::new(&gpu))
        };
        self.state = Some(State {
            window,
            gpu,
            sim,
            overlay,
            input: Input::default(),
            last: Instant::now(),
            fps: 60.0,
            screenshot: false,
            capture,
            shot_idx: 0,
            phase_start: Instant::now(),
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
                        KeyCode::KeyP => st.screenshot = true,
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

                // In capture mode, once the current backend has simulated for its
                // warmup window, request a screenshot of this frame.
                let shoot_now = st.capture
                    && st.shot_idx < CAPTURE_PLAN.len()
                    && st.phase_start.elapsed().as_secs_f32() >= CAPTURE_PLAN[st.shot_idx].1;
                if shoot_now {
                    st.screenshot = true;
                }

                st.sim.update(&st.gpu, dt, &st.input);
                st.input.end_frame();

                if let Some(frame) = st.gpu.acquire() {
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    st.sim.render(&st.gpu, &view);

                    // Save a clean PNG of the rendered fluid (before the egui
                    // panel is composited on top) when the user presses P.
                    if st.screenshot {
                        st.screenshot = false;
                        let slug: String = st
                            .sim
                            .name()
                            .chars()
                            .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
                            .collect();
                        let dir = std::path::Path::new("assets");
                        std::fs::create_dir_all(dir).ok();
                        let path = dir.join(format!("ripple-{slug}.png"));
                        st.gpu.save_png(&frame.texture, &path);
                        eprintln!("saved screenshot to {}", path.display());
                    }

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
                // Advance the capture plan only once the shot was actually saved
                // (a skipped frame leaves the request pending for next time).
                if shoot_now && !st.screenshot {
                    st.shot_idx += 1;
                    if st.shot_idx >= CAPTURE_PLAN.len() {
                        event_loop.exit();
                    } else {
                        st.sim = make_backend(CAPTURE_PLAN[st.shot_idx].0, &st.gpu);
                        st.phase_start = Instant::now();
                        st.last = Instant::now();
                    }
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
