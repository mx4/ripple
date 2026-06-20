//! 2D Smoothed-Particle Hydrodynamics (SPH) fluid solver.
//!
//! Weakly-compressible SPH after Müller et al. 2003 ("Particle-Based Fluid
//! Simulation for Interactive Applications"). The fluid is a cloud of particles;
//! pressure pushes them apart toward a rest density, viscosity smooths their
//! relative motion, and gravity pulls them down. Boundaries are handled by
//! `Bounds` (see below) so the container shape is decoupled from the physics.
//!
//! Everything is in *pixel* units so positions map straight to the screen.

use glam::{vec2, Vec2};
use rayon::prelude::*;

// --- Solver constants (tuned together; change with care) -------------------
/// Kernel radius: a particle only interacts with neighbours within `H` pixels.
pub const H: f32 = 16.0;
const HSQ: f32 = H * H;
/// Mass of every particle (uniform). The absolute value is unimportant: density
/// scales with it, and pressure acceleration depends only on *relative*
/// compression × stiffness, so this just sets the (arbitrary) density units.
const MASS: f32 = 2.5;
/// Default stiffness of the pressure response (linear equation of state). This
/// is the main knob trading incompressibility (high) against timestep (low).
/// Chosen from the `sweep` test: peak compression ~2.2x (steady-state far
/// lower), calm rebound speeds, stable even at dt = 0.0008.
const DEFAULT_STIFFNESS: f32 = 2_000_000.0;
/// Viscosity coefficient — higher = thicker, more syrup-like fluid.
const VISC: f32 = 200.0;
/// Restitution at the walls: how much normal velocity survives a bounce.
const BOUND_DAMPING: f32 = 0.4;
/// Safety net: hard cap on particle speed (px/s) to keep the sim finite.
const MAX_SPEED: f32 = 1500.0;
/// Tiny density floor to avoid divide-by-zero for isolated particles. Must stay
/// well below the (small) measured rest density so it never distorts dynamics.
const MIN_RHO: f32 = 1e-6;

/// Radius used when drawing a particle, and as the wall keep-out distance.
pub const PARTICLE_RADIUS: f32 = H * 0.5;

/// The tuned SPH constants, bundled so the GPU backend can use the exact same
/// values (single source of truth). Kernel factors depend only on `H`.
#[derive(Clone, Copy, Debug)]
pub struct SphConstants {
    pub h: f32,
    pub mass: f32,
    pub poly6: f32,
    pub spiky_grad: f32,
    pub visc_lap: f32,
    pub visc: f32,
    pub stiffness: f32,
    pub bound_damping: f32,
    pub particle_radius: f32,
    pub max_speed: f32,
}

/// The shipped solver constants (matches the CPU solver and its tuning).
pub fn sph_constants() -> SphConstants {
    let pi = std::f32::consts::PI;
    SphConstants {
        h: H,
        mass: MASS,
        poly6: 4.0 / (pi * H.powi(8)),
        spiky_grad: 30.0 / (pi * H.powi(5)),
        visc_lap: 40.0 / (pi * H.powi(5)),
        visc: VISC,
        stiffness: DEFAULT_STIFFNESS,
        bound_damping: BOUND_DAMPING,
        particle_radius: PARTICLE_RADIUS,
        max_speed: MAX_SPEED,
    }
}

/// Container shape the fluid is confined to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    Rect,
    Circle,
}

/// The container the fluid lives in. Decoupled from the solver so new shapes
/// only need a new `resolve` branch.
#[derive(Clone, Copy)]
pub struct Bounds {
    pub w: f32,
    pub h: f32,
    pub shape: Shape,
}

impl Bounds {
    /// Push a particle back inside the container and reflect its velocity.
    fn resolve(&self, pos: &mut Vec2, vel: &mut Vec2) {
        let pad = PARTICLE_RADIUS;
        match self.shape {
            Shape::Rect => {
                if pos.x < pad {
                    pos.x = pad;
                    vel.x *= -BOUND_DAMPING;
                } else if pos.x > self.w - pad {
                    pos.x = self.w - pad;
                    vel.x *= -BOUND_DAMPING;
                }
                if pos.y < pad {
                    pos.y = pad;
                    vel.y *= -BOUND_DAMPING;
                } else if pos.y > self.h - pad {
                    pos.y = self.h - pad;
                    vel.y *= -BOUND_DAMPING;
                }
            }
            Shape::Circle => {
                let center = vec2(self.w * 0.5, self.h * 0.5);
                let radius = self.w.min(self.h) * 0.5 - pad;
                let d = *pos - center;
                let dist = d.length();
                if dist > radius && dist > 0.0 {
                    let n = d / dist; // outward normal
                    *pos = center + n * radius;
                    let vn = vel.dot(n);
                    if vn > 0.0 {
                        // Reflect & damp the outward (normal) component only.
                        *vel -= (1.0 + BOUND_DAMPING) * vn * n;
                    }
                }
            }
        }
    }
}

/// SPH fluid stored as struct-of-arrays (cache friendly, borrow-checker happy).
pub struct Sim {
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    force: Vec<Vec2>, // pressure + viscosity (gravity added at integration time)
    rho: Vec<f32>,
    pressure: Vec<f32>,

    /// Rest density, measured from the initial packing so the fluid starts calm.
    rest_dens: f32,
    /// Pressure stiffness (gas constant of the equation of state).
    pub stiffness: f32,

    // Precomputed 2D kernel normalisation factors.
    poly6: f32,
    spiky_grad: f32,
    visc_lap: f32,

    // Reusable uniform-grid acceleration structure for neighbour search.
    grid: Vec<Vec<usize>>,
    cols: usize,
    rows: usize,
}

impl Sim {
    /// Fill the lower portion of a `w` x `h` box with a block of particles.
    pub fn new(w: f32, h: f32) -> Self {
        let pi = std::f32::consts::PI;
        let mut sim = Sim {
            pos: Vec::new(),
            vel: Vec::new(),
            force: Vec::new(),
            rho: Vec::new(),
            pressure: Vec::new(),
            rest_dens: 1.0,
            stiffness: DEFAULT_STIFFNESS,
            // 2D kernel normalisations. `spiky_grad` is the positive magnitude
            // of the spiky gradient coefficient; the repulsive direction is
            // applied explicitly in `compute_forces`.
            poly6: 4.0 / (pi * H.powi(8)),
            spiky_grad: 30.0 / (pi * H.powi(5)),
            visc_lap: 40.0 / (pi * H.powi(5)),
            grid: Vec::new(),
            cols: 0,
            rows: 0,
        };
        sim.spawn_block(w, h);
        // Calibrate rest density from the initial (roughly at-rest) packing so
        // the interior starts near zero pressure instead of exploding.
        let bounds = Bounds { w, h, shape: Shape::Rect };
        sim.build_grid(&bounds);
        sim.compute_density_pressure();
        // Interior particles have the most neighbours → highest density. Use
        // that as the rest density so the packed interior starts at ~zero
        // pressure (surface particles, being sparser, stay slightly negative
        // and are clamped to zero). The absolute value is in arbitrary units.
        let max_rho = sim.rho.iter().cloned().fold(0.0_f32, f32::max);
        sim.rest_dens = max_rho.max(MIN_RHO);
        sim
    }

    /// Reset particles to the starting block for the current container size.
    pub fn reset(&mut self, w: f32, h: f32) {
        self.pos.clear();
        self.vel.clear();
        self.spawn_block(w, h);
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Rest density measured from the initial packing — needed by the GPU
    /// backend so both solvers start at the same near-zero-pressure state.
    pub fn rest_density(&self) -> f32 {
        self.rest_dens
    }

    /// Initial particle positions as flat `[x, y]` pairs, for uploading to the
    /// GPU (`Sim::new` already does the spawn + rest-density calibration).
    pub fn positions_xy(&self) -> Vec<[f32; 2]> {
        self.pos.iter().map(|p| [p.x, p.y]).collect()
    }

    /// Add the same velocity to every particle — a "shake" / impulse.
    pub fn add_impulse(&mut self, dv: Vec2) {
        for v in &mut self.vel {
            *v += dv;
        }
    }

    fn spawn_block(&mut self, w: f32, h: f32) {
        let spacing = H * 0.6;
        // A block occupying the left ~45% and bottom ~70% of the box.
        let x0 = w * 0.08;
        let x1 = w * 0.45;
        let y0 = h * 0.30;
        let y1 = h * 0.92;
        let mut y = y0;
        let mut row = 0;
        while y < y1 {
            // Stagger every other row for a denser, more natural packing.
            let mut x = x0 + if row % 2 == 0 { 0.0 } else { spacing * 0.5 };
            while x < x1 {
                self.pos.push(vec2(x, y));
                self.vel.push(Vec2::ZERO);
                self.force.push(Vec2::ZERO);
                self.rho.push(0.0);
                self.pressure.push(0.0);
                x += spacing;
            }
            y += spacing;
            row += 1;
        }
    }

    /// Advance the simulation by one timestep `dt` under `gravity` (px/s²).
    pub fn step(&mut self, dt: f32, gravity: Vec2, bounds: &Bounds) {
        self.build_grid(bounds);
        self.compute_density_pressure();
        self.compute_forces();
        self.integrate(dt, gravity, bounds);
    }

    // --- neighbour search ---------------------------------------------------
    fn build_grid(&mut self, bounds: &Bounds) {
        let cell = H;
        let cols = (bounds.w / cell).ceil().max(1.0) as usize + 1;
        let rows = (bounds.h / cell).ceil().max(1.0) as usize + 1;
        if cols != self.cols || rows != self.rows || self.grid.len() != cols * rows {
            self.grid = vec![Vec::new(); cols * rows];
            self.cols = cols;
            self.rows = rows;
        } else {
            for cell in &mut self.grid {
                cell.clear();
            }
        }
        for (i, p) in self.pos.iter().enumerate() {
            let (cx, cy) = cell_of(*p, cols, rows);
            self.grid[cy * cols + cx].push(i);
        }
    }

    // --- SPH passes ---------------------------------------------------------
    fn compute_density_pressure(&mut self) {
        // Destructure for disjoint borrows: read pos/grid (shared) while writing
        // rho/pressure (one slot per particle, so threads never collide).
        let Sim {
            pos,
            rho,
            pressure,
            grid,
            cols,
            rows,
            poly6,
            stiffness,
            rest_dens,
            ..
        } = self;
        let (cols, rows) = (*cols, *rows);
        let (poly6, stiffness, rest_dens) = (*poly6, *stiffness, *rest_dens);
        let pos: &[Vec2] = pos;
        let grid: &[Vec<usize>] = grid;

        rho.par_iter_mut()
            .zip(pressure.par_iter_mut())
            .enumerate()
            .for_each(|(i, (rho_out, p_out))| {
                let pi = pos[i];
                let mut r = 0.0;
                for_neighbors(pi, grid, cols, rows, |j| {
                    let r2 = (pos[j] - pi).length_squared();
                    if r2 < HSQ {
                        r += MASS * poly6 * (HSQ - r2).powi(3);
                    }
                });
                *rho_out = r;
                // Clamp to >= 0: never let the fluid pull itself together
                // (avoids the classic SPH "tensile instability" clumping).
                *p_out = (stiffness * (r - rest_dens)).max(0.0);
            });
    }

    fn compute_forces(&mut self) {
        let Sim {
            pos,
            vel,
            force,
            rho,
            pressure,
            grid,
            cols,
            rows,
            spiky_grad,
            visc_lap,
            ..
        } = self;
        let (cols, rows) = (*cols, *rows);
        let (spiky_grad, visc_lap) = (*spiky_grad, *visc_lap);
        let pos: &[Vec2] = pos;
        let vel: &[Vec2] = vel;
        let rho: &[f32] = rho;
        let pressure: &[f32] = pressure;
        let grid: &[Vec<usize>] = grid;

        force.par_iter_mut().enumerate().for_each(|(i, f_out)| {
            let pi = pos[i];
            let vi = vel[i];
            let mut fpress = Vec2::ZERO;
            let mut fvisc = Vec2::ZERO;
            for_neighbors(pi, grid, cols, rows, |j| {
                if j == i {
                    return;
                }
                let rij = pos[j] - pi;
                let r = rij.length();
                if r < H && r > 0.0 {
                    let dir = rij / r; // points from i toward neighbour j
                    let rho_j = rho[j].max(MIN_RHO);
                    // Pressure repels: force points away from j (-dir).
                    fpress += -dir * MASS * (pressure[i] + pressure[j]) / (2.0 * rho_j)
                        * spiky_grad
                        * (H - r).powi(2);
                    fvisc += VISC * MASS * (vel[j] - vi) / rho_j * visc_lap * (H - r);
                }
            });
            *f_out = fpress + fvisc;
        });
    }

    fn integrate(&mut self, dt: f32, gravity: Vec2, bounds: &Bounds) {
        let Sim {
            pos, vel, force, rho, ..
        } = self;
        let force: &[Vec2] = force;
        let rho: &[f32] = rho;

        pos.par_iter_mut()
            .zip(vel.par_iter_mut())
            .enumerate()
            .for_each(|(i, (p, v))| {
                let r = rho[i].max(MIN_RHO);
                // a = (pressure + viscosity) / rho + gravity
                let accel = force[i] / r + gravity;
                *v += dt * accel;
                let speed = v.length();
                if speed > MAX_SPEED {
                    *v *= MAX_SPEED / speed;
                }
                *p += dt * *v;
                bounds.resolve(p, v);
            });
    }
}

// --- neighbour search (free functions so the SPH passes can be parallelised
// with rayon without borrowing all of `self`) -------------------------------

#[inline]
fn cell_of(p: Vec2, cols: usize, rows: usize) -> (usize, usize) {
    let cx = (p.x / H).clamp(0.0, (cols - 1) as f32) as usize;
    let cy = (p.y / H).clamp(0.0, (rows - 1) as f32) as usize;
    (cx, cy)
}

/// Visit the index of every particle in the 3x3 block of cells around `p`.
#[inline]
fn for_neighbors(p: Vec2, grid: &[Vec<usize>], cols: usize, rows: usize, mut f: impl FnMut(usize)) {
    let (cx, cy) = cell_of(p, cols, rows);
    let gx0 = cx.saturating_sub(1);
    let gy0 = cy.saturating_sub(1);
    let gx1 = (cx + 1).min(cols - 1);
    let gy1 = (cy + 1).min(rows - 1);
    for gy in gy0..=gy1 {
        for gx in gx0..=gx1 {
            for &j in &grid[gy * cols + gx] {
                f(j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Diag {
        finite: bool,
        escaped: bool,
        max_speed: f32,
        max_compression: f32, // peak rho / rest_dens
        capped_frac: f32,     // fraction of particles pegged at MAX_SPEED
        avg_y: f32,
    }

    /// Run for `sim_time` seconds of simulated time and report diagnostics.
    fn run(stiffness: f32, dt: f32, sim_time: f32) -> Diag {
        let (w, h) = (900.0, 600.0);
        let mut sim = Sim::new(w, h);
        sim.stiffness = stiffness;
        let bounds = Bounds { w, h, shape: Shape::Rect };
        let gravity = vec2(0.0, 1200.0);

        let steps = (sim_time / dt) as usize;
        let mut max_speed = 0.0_f32;
        let mut max_compression = 0.0_f32;
        for _ in 0..steps {
            sim.step(dt, gravity, &bounds);
            for v in &sim.vel {
                max_speed = max_speed.max(v.length());
            }
            for &r in &sim.rho {
                max_compression = max_compression.max(r / sim.rest_dens);
            }
        }

        let finite = sim.pos.iter().all(|p| p.is_finite());
        let escaped = sim
            .pos
            .iter()
            .any(|p| p.x < -2.0 || p.x > w + 2.0 || p.y < -2.0 || p.y > h + 2.0);
        let capped = sim
            .vel
            .iter()
            .filter(|v| v.length() >= MAX_SPEED - 1.0)
            .count();
        let avg_y = sim.pos.iter().map(|p| p.y).sum::<f32>() / sim.len() as f32;
        Diag {
            finite,
            escaped,
            max_speed,
            max_compression,
            capped_frac: capped as f32 / sim.len() as f32,
            avg_y,
        }
    }

    /// Diagnostic sweep (run with `--nocapture`) to choose stiffness/dt. A good
    /// liquid: finite, not escaped, settles low (high avg_y), compression near
    /// 1, and very few particles pegged at the speed cap.
    #[test]
    #[ignore]
    fn sweep() {
        println!("\n stiffness      dt   finite escaped maxspd  maxcomp capped%  avg_y");
        for &stiffness in &[50_000.0, 150_000.0, 250_000.0, 500_000.0, 1_000_000.0] {
            for &dt in &[0.0008_f32, 0.0004, 0.0002] {
                let d = run(stiffness, dt, 4.0);
                println!(
                    "{:>10.0} {:>7.4}   {:>5} {:>6}  {:>6.0}  {:>6.2}  {:>5.1}  {:>6.1}",
                    stiffness,
                    dt,
                    d.finite,
                    d.escaped,
                    d.max_speed,
                    d.max_compression,
                    d.capped_frac * 100.0,
                    d.avg_y
                );
            }
        }
        println!();
    }

    /// Guard against regressions with the shipped defaults: the fluid must stay
    /// finite, in-bounds, behave like a liquid (only mildly compressible), and
    /// settle toward the floor under gravity.
    #[test]
    fn default_params_are_stable_liquid() {
        let d = run(DEFAULT_STIFFNESS, 0.0004, 5.0);
        assert!(d.finite, "sim produced NaN/inf");
        assert!(!d.escaped, "particles escaped the container");
        // Peak compression is a brief impact transient; a healthy liquid stays
        // well under ~3x. (The pressure-sign regression produced 100x+.)
        assert!(
            d.max_compression < 3.0,
            "fluid over-compressed (max {:.2}x rest) — not liquid-like",
            d.max_compression
        );
        assert!(
            d.capped_frac < 0.02,
            "too many particles pegged at speed cap ({:.1}%) — unstable",
            d.capped_frac * 100.0
        );
        assert!(d.avg_y > 600.0 * 0.55, "fluid didn't settle (avg_y {:.1})", d.avg_y);
        println!(
            "ok: maxspd={:.0} maxcomp={:.2} capped={:.2}% avg_y={:.1}",
            d.max_speed,
            d.max_compression,
            d.capped_frac * 100.0,
            d.avg_y
        );
    }

    /// Throughput benchmark (run with `--ignored --nocapture`). Compare:
    ///   cargo test --release bench -- --ignored --nocapture
    ///   RAYON_NUM_THREADS=1 cargo test --release bench -- --ignored --nocapture
    /// to see the speedup from parallelising the SPH passes.
    #[test]
    #[ignore]
    fn bench() {
        // A bigger box → more particles, where parallelism actually pays off.
        let (w, h) = (1800.0, 1200.0);
        let mut sim = Sim::new(w, h);
        let bounds = Bounds { w, h, shape: Shape::Rect };
        let gravity = vec2(0.0, 1200.0);
        let dt = 0.0008;

        for _ in 0..50 {
            sim.step(dt, gravity, &bounds); // warm up / let the grid settle
        }

        let steps = 1000;
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            sim.step(dt, gravity, &bounds);
        }
        let secs = t0.elapsed().as_secs_f64();

        let threads = rayon::current_num_threads();
        let n = sim.len();
        println!(
            "bench: threads={threads} n={n} steps={steps} \
             time={secs:.3}s  {:.0} steps/s  {:.1} M particle-steps/s",
            steps as f64 / secs,
            (n as f64 * steps as f64) / secs / 1e6
        );
    }
}
