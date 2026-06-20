//! GPU backend (wgpu): a shared `Gpu` context, a `Simulation` trait that fluid
//! models implement, and the building blocks (compute solver, renderers) they
//! use. The app (`src/bin/gpu.rs`) owns the context and a `Box<dyn Simulation>`.

mod backend;
mod context;
mod cpu_backends;
mod field;
mod particles;
mod render;
mod sim;
mod smoke_gpu;
mod sph_backend;
mod ui;

pub use backend::{Input, Simulation};
pub use context::Gpu;
pub use cpu_backends::{FlipBackend, SmokeBackend, SphCpuBackend};
pub use field::FieldRenderer;
pub use particles::ParticleRenderer;
pub use render::{RenderMode, Renderer};
pub use sim::{headless_device, GpuSim};
pub use smoke_gpu::{GpuSmoke, GpuSmokeBackend};
pub use sph_backend::SphBackend;
pub use ui::EguiOverlay;
