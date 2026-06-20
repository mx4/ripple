//! Fluid simulation library: a CPU SPH solver (`sim`) and an optional GPU
//! compute backend (`gpu`, behind the `gpu` feature).

pub mod flip;
pub mod gpu;
pub mod sim;
pub mod smoke;
