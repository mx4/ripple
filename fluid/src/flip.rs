//! 2D FLIP/PIC water — a hybrid liquid solver with a free surface.
//!
//! Particles carry the fluid (and, by where they are, define the liquid region);
//! the heavy lifting — making the velocity field divergence-free — happens on a
//! staggered MAC grid, the same idea the smoke backend uses. Each step:
//!
//!   1. integrate particles (gravity + move + wall collisions)
//!   2. P2G: splat particle velocities onto the grid; mark fluid/air/solid cells
//!   3. solve incompressibility on the grid (Gauss-Seidel; only in fluid cells,
//!      air left free = the free surface, no flow through solid walls)
//!   4. G2P: read grid velocities back to particles, blending PIC (grid) with
//!      FLIP (grid *change*) for lively-but-stable motion
//!
//! Grid: `nx x ny` cells, cell size `h`. Velocities live on cell faces
//! (staggered): `u` (x) on vertical faces `(nx+1) x ny`, `v` (y) on horizontal
//! faces `nx x (ny+1)`. `y` increases upward; gravity points down.
//!
//! Pure math (no rendering deps) so it can be unit-tested headlessly.

/// Tunable parameters (separate from the heavy buffers in `FlipSim`).
#[derive(Clone, Copy)]
pub struct Flip {
    /// Gravity (world units/s²); default points down.
    pub gravity: [f32; 2],
    /// PIC/FLIP blend: 0 = pure PIC (damped), 1 = pure FLIP (lively, noisy).
    pub flip_ratio: f32,
    /// Over-relaxation for the pressure Gauss-Seidel (~1.9).
    pub over_relax: f32,
    /// Pressure solver iterations per step.
    pub iters: usize,
}

/// The full simulation: config + particle and grid state.
pub struct FlipSim {
    cfg: Flip,
    nx: usize,
    ny: usize,
    h: f32,

    pos: Vec<[f32; 2]>,
    vel: Vec<[f32; 2]>,

    u: Vec<f32>,
    v: Vec<f32>,
    u_prev: Vec<f32>,
    v_prev: Vec<f32>,
    wu: Vec<f32>,
    wv: Vec<f32>,

    s: Vec<f32>,          // 0 = solid cell, 1 = otherwise (per cell)
    is_fluid: Vec<bool>,  // cell currently contains particles
}

#[inline]
fn idx(i: usize, j: usize, cols: usize) -> usize {
    i + cols * j
}

impl FlipSim {
    /// Create an `nx x ny` grid (cell size `h`) and fill the given world-space
    /// rectangle with a jittered block of particles (~4 per cell).
    pub fn new(nx: usize, ny: usize, h: f32) -> Self {
        let mut s = vec![1.0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                    s[idx(i, j, nx)] = 0.0; // solid border
                }
            }
        }
        let mut sim = FlipSim {
            cfg: Flip {
                gravity: [0.0, -9.0],
                flip_ratio: 0.9,
                over_relax: 1.9,
                iters: 50,
            },
            nx,
            ny,
            h,
            pos: Vec::new(),
            vel: Vec::new(),
            u: vec![0.0; (nx + 1) * ny],
            v: vec![0.0; nx * (ny + 1)],
            u_prev: vec![0.0; (nx + 1) * ny],
            v_prev: vec![0.0; nx * (ny + 1)],
            wu: vec![0.0; (nx + 1) * ny],
            wv: vec![0.0; nx * (ny + 1)],
            s,
            is_fluid: vec![false; nx * ny],
        };
        sim.spawn_block();
        sim
    }

    pub fn config(&mut self) -> &mut Flip {
        &mut self.cfg
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn domain(&self) -> (f32, f32) {
        (self.nx as f32 * self.h, self.ny as f32 * self.h)
    }

    pub fn particles(&self) -> impl Iterator<Item = ([f32; 2], [f32; 2])> + '_ {
        self.pos.iter().copied().zip(self.vel.iter().copied())
    }

    pub fn reset(&mut self) {
        self.pos.clear();
        self.vel.clear();
        self.spawn_block();
    }

    /// Add the same velocity to every particle — a global "shake" / jolt.
    pub fn add_impulse(&mut self, dvx: f32, dvy: f32) {
        for vel in &mut self.vel {
            vel[0] += dvx;
            vel[1] += dvy;
        }
    }

    /// Add a velocity impulse to particles within `radius` of a world point.
    pub fn push(&mut self, x: f32, y: f32, dvx: f32, dvy: f32, radius: f32) {
        let r2 = radius * radius;
        for (p, vel) in self.pos.iter().zip(self.vel.iter_mut()) {
            let dx = p[0] - x;
            let dy = p[1] - y;
            if dx * dx + dy * dy < r2 {
                vel[0] += dvx;
                vel[1] += dvy;
            }
        }
    }

    fn spawn_block(&mut self) {
        // Fill the left ~40% and bottom ~60% of the (non-solid) domain, ~4 per cell.
        let (w, hgt) = self.domain();
        let x0 = w * 0.06;
        let x1 = w * 0.45;
        let y0 = hgt * 0.06;
        let y1 = hgt * 0.6;
        let step = self.h * 0.5; // 2x2 particles per cell
        let mut y = y0;
        let mut row = 0;
        while y < y1 {
            let mut x = x0 + if row % 2 == 0 { 0.0 } else { step * 0.5 };
            while x < x1 {
                // small jitter to avoid a perfect lattice
                let jx = (hash(self.pos.len() as u32) - 0.5) * step * 0.3;
                let jy = (hash(self.pos.len() as u32 ^ 0x9e37) - 0.5) * step * 0.3;
                self.pos.push([x + jx, y + jy]);
                self.vel.push([0.0, 0.0]);
                x += step;
            }
            y += step;
            row += 1;
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.integrate_particles(dt);
        self.p2g();
        self.solve_incompressibility();
        self.g2p();
    }

    fn integrate_particles(&mut self, dt: f32) {
        let (g0, g1) = (self.cfg.gravity[0], self.cfg.gravity[1]);
        let eps = 1e-3 * self.h;
        let xmin = self.h + eps;
        let xmax = (self.nx as f32 - 1.0) * self.h - eps;
        let ymin = self.h + eps;
        let ymax = (self.ny as f32 - 1.0) * self.h - eps;
        for (p, vel) in self.pos.iter_mut().zip(self.vel.iter_mut()) {
            vel[0] += g0 * dt;
            vel[1] += g1 * dt;
            p[0] += vel[0] * dt;
            p[1] += vel[1] * dt;
            if p[0] < xmin {
                p[0] = xmin;
                vel[0] = 0.0;
            } else if p[0] > xmax {
                p[0] = xmax;
                vel[0] = 0.0;
            }
            if p[1] < ymin {
                p[1] = ymin;
                vel[1] = 0.0;
            } else if p[1] > ymax {
                p[1] = ymax;
                vel[1] = 0.0;
            }
        }
    }

    fn p2g(&mut self) {
        self.u.iter_mut().for_each(|x| *x = 0.0);
        self.v.iter_mut().for_each(|x| *x = 0.0);
        self.wu.iter_mut().for_each(|x| *x = 0.0);
        self.wv.iter_mut().for_each(|x| *x = 0.0);

        let (nx, ny, h) = (self.nx, self.ny, self.h);
        for (p, vel) in self.pos.iter().zip(self.vel.iter()) {
            let gx = p[0] / h;
            let gy = p[1] / h;
            // u faces at (i, j+0.5)  -> grid coords (gx, gy - 0.5), dims (nx+1, ny)
            splat(&mut self.u, &mut self.wu, nx + 1, ny, gx, gy - 0.5, vel[0]);
            // v faces at (i+0.5, j)  -> grid coords (gx - 0.5, gy), dims (nx, ny+1)
            splat(&mut self.v, &mut self.wv, nx, ny + 1, gx - 0.5, gy, vel[1]);
        }
        for k in 0..self.u.len() {
            if self.wu[k] > 0.0 {
                self.u[k] /= self.wu[k];
            }
        }
        for k in 0..self.v.len() {
            if self.wv[k] > 0.0 {
                self.v[k] /= self.wv[k];
            }
        }
        self.u_prev.copy_from_slice(&self.u);
        self.v_prev.copy_from_slice(&self.v);

        // Mark fluid cells (interior cells containing a particle).
        self.is_fluid.iter_mut().for_each(|c| *c = false);
        for p in &self.pos {
            let ci = (p[0] / h) as usize;
            let cj = (p[1] / h) as usize;
            if ci < nx && cj < ny && self.s[idx(ci, cj, nx)] > 0.0 {
                self.is_fluid[idx(ci, cj, nx)] = true;
            }
        }
    }

    fn solve_incompressibility(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let or = self.cfg.over_relax;
        for _ in 0..self.cfg.iters {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    if !self.is_fluid[idx(i, j, nx)] {
                        continue;
                    }
                    let sx0 = self.s[idx(i - 1, j, nx)];
                    let sx1 = self.s[idx(i + 1, j, nx)];
                    let sy0 = self.s[idx(i, j - 1, nx)];
                    let sy1 = self.s[idx(i, j + 1, nx)];
                    let sum = sx0 + sx1 + sy0 + sy1;
                    if sum == 0.0 {
                        continue;
                    }
                    let div = self.u[idx(i + 1, j, nx + 1)] - self.u[idx(i, j, nx + 1)]
                        + self.v[idx(i, j + 1, nx)]
                        - self.v[idx(i, j, nx)];
                    let p = -div / sum * or;
                    self.u[idx(i, j, nx + 1)] -= sx0 * p;
                    self.u[idx(i + 1, j, nx + 1)] += sx1 * p;
                    self.v[idx(i, j, nx)] -= sy0 * p;
                    self.v[idx(i, j + 1, nx)] += sy1 * p;
                }
            }
        }
    }

    fn g2p(&mut self) {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let alpha = self.cfg.flip_ratio;
        for (p, vel) in self.pos.iter().zip(self.vel.iter_mut()) {
            let gx = p[0] / h;
            let gy = p[1] / h;
            let ux_new = sample(&self.u, nx + 1, ny, gx, gy - 0.5);
            let ux_old = sample(&self.u_prev, nx + 1, ny, gx, gy - 0.5);
            let uy_new = sample(&self.v, nx, ny + 1, gx - 0.5, gy);
            let uy_old = sample(&self.v_prev, nx, ny + 1, gx - 0.5, gy);

            let pic_x = ux_new;
            let pic_y = uy_new;
            let flip_x = vel[0] + (ux_new - ux_old);
            let flip_y = vel[1] + (uy_new - uy_old);
            vel[0] = alpha * flip_x + (1.0 - alpha) * pic_x;
            vel[1] = alpha * flip_y + (1.0 - alpha) * pic_y;
        }
    }
}

/// Cheap deterministic hash -> [0,1), for spawn jitter.
fn hash(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

/// Bilinearly scatter `val` into a `cols x rows` grid at continuous `(gx, gy)`.
fn splat(field: &mut [f32], wt: &mut [f32], cols: usize, rows: usize, gx: f32, gy: f32, val: f32) {
    let gx = gx.clamp(0.0, cols as f32 - 1.0 - 1e-4);
    let gy = gy.clamp(0.0, rows as f32 - 1.0 - 1e-4);
    let i0 = gx as usize;
    let j0 = gy as usize;
    let fx = gx - i0 as f32;
    let fy = gy - j0 as f32;
    let w = [
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ];
    let k = [
        idx(i0, j0, cols),
        idx(i0 + 1, j0, cols),
        idx(i0, j0 + 1, cols),
        idx(i0 + 1, j0 + 1, cols),
    ];
    for n in 0..4 {
        field[k[n]] += w[n] * val;
        wt[k[n]] += w[n];
    }
}

/// Bilinearly sample a `cols x rows` grid at continuous `(gx, gy)`.
fn sample(field: &[f32], cols: usize, rows: usize, gx: f32, gy: f32) -> f32 {
    let gx = gx.clamp(0.0, cols as f32 - 1.0 - 1e-4);
    let gy = gy.clamp(0.0, rows as f32 - 1.0 - 1e-4);
    let i0 = gx as usize;
    let j0 = gy as usize;
    let fx = gx - i0 as f32;
    let fy = gy - j0 as f32;
    (1.0 - fx) * (1.0 - fy) * field[idx(i0, j0, cols)]
        + fx * (1.0 - fy) * field[idx(i0 + 1, j0, cols)]
        + (1.0 - fx) * fy * field[idx(i0, j0 + 1, cols)]
        + fx * fy * field[idx(i0 + 1, j0 + 1, cols)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drop a block of water and check it stays finite, in-bounds, and settles
    /// toward the floor (low y, since gravity points down) — the headless guard.
    #[test]
    fn water_settles_and_is_stable() {
        let (nx, ny, h) = (64, 64, 1.0);
        let mut sim = FlipSim::new(nx, ny, h);
        assert!(sim.len() > 500, "expected a decent block of particles");
        let (w, hgt) = sim.domain();

        let dt = 1.0 / 60.0;
        let mut max_speed = 0.0f32;
        for _ in 0..600 {
            sim.step(dt);
            for (_, v) in sim.particles() {
                max_speed = max_speed.max((v[0] * v[0] + v[1] * v[1]).sqrt());
            }
        }

        let mut avg_y = 0.0f32;
        for (p, _) in sim.particles() {
            assert!(p[0].is_finite() && p[1].is_finite(), "particle NaN/inf");
            assert!(p[0] > 0.0 && p[0] < w, "escaped x: {}", p[0]);
            assert!(p[1] > 0.0 && p[1] < hgt, "escaped y: {}", p[1]);
            avg_y += p[1];
        }
        avg_y /= sim.len() as f32;
        // Water should pool low under gravity (well below mid-height).
        assert!(avg_y < hgt * 0.4, "water did not settle (avg_y {avg_y})");
        // It should also spread out along the floor, not collapse to a column.
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        for (p, _) in sim.particles() {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
        }
        assert!(max_x - min_x > w * 0.3, "water didn't spread (width {})", max_x - min_x);
        println!(
            "flip ok: n={} max_speed={:.1} avg_y={:.1} width={:.1}",
            sim.len(),
            max_speed,
            avg_y,
            max_x - min_x
        );
    }
}
