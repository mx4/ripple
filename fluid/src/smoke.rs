//! 2D Eulerian smoke solver — Jos Stam's "Stable Fluids" / "Real-Time Fluid
//! Dynamics for Games". Unlike the SPH backend (particles, a liquid with a free
//! surface), this models a *gas* on a fixed grid: velocity + density fields, a
//! semi-Lagrangian advection, and a pressure projection that keeps the velocity
//! divergence-free. On top of the base method it adds **buoyancy** (smoke rises)
//! and **vorticity confinement** (puts back the small curls numerical diffusion
//! eats — the wispy character of smoke).
//!
//! Grid is `n x n` interior cells with a one-cell border, stored row-major in
//! `(n+2)*(n+2)` arrays. Index `(i, j)`, with `j` increasing upward.
//!
//! Pure math (no rendering deps) so it can be unit-tested headlessly.

#[inline]
fn ix(i: usize, j: usize, n: usize) -> usize {
    i + (n + 2) * j
}

pub struct Smoke {
    n: usize,
    // velocity (u = x, v = y) and its per-frame source/scratch
    u: Vec<f32>,
    v: Vec<f32>,
    u0: Vec<f32>,
    v0: Vec<f32>,
    // density (smoke concentration) and its per-frame source/scratch
    dens: Vec<f32>,
    dens0: Vec<f32>,
    // scratch for vorticity confinement
    curl: Vec<f32>,

    /// Diffusion of the smoke density (usually ~0 — smoke advects, barely diffuses).
    pub diff: f32,
    /// Kinematic viscosity of the velocity field (usually ~0 for smoke).
    pub visc: f32,
    /// Buoyancy: upward force per unit density (makes smoke rise).
    pub buoyancy: f32,
    /// Per-second fraction by which density fades (so smoke dissipates).
    pub dissipation: f32,
    /// Vorticity-confinement strength (0 disables; adds swirly detail).
    pub vorticity: f32,
}

impl Smoke {
    pub fn new(n: usize) -> Self {
        let size = (n + 2) * (n + 2);
        Smoke {
            n,
            u: vec![0.0; size],
            v: vec![0.0; size],
            u0: vec![0.0; size],
            v0: vec![0.0; size],
            dens: vec![0.0; size],
            dens0: vec![0.0; size],
            curl: vec![0.0; size],
            diff: 0.0,
            visc: 0.0,
            buoyancy: 1.0,
            dissipation: 0.4,
            vorticity: 3.0,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Smoke density at an interior cell (`i, j` in `1..=n`).
    pub fn density_at(&self, i: usize, j: usize) -> f32 {
        self.dens[ix(i, j, self.n)]
    }

    /// Inject smoke at interior cell `(i, j)` (accumulated as a per-frame source).
    pub fn add_density(&mut self, i: usize, j: usize, amount: f32) {
        if i >= 1 && i <= self.n && j >= 1 && j <= self.n {
            let k = ix(i, j, self.n);
            self.dens0[k] += amount;
        }
    }

    /// Inject velocity at interior cell `(i, j)` (per-frame force source).
    pub fn add_velocity(&mut self, i: usize, j: usize, du: f32, dv: f32) {
        if i >= 1 && i <= self.n && j >= 1 && j <= self.n {
            let k = ix(i, j, self.n);
            self.u0[k] += du;
            self.v0[k] += dv;
        }
    }

    pub fn reset(&mut self) {
        for buf in [
            &mut self.u,
            &mut self.v,
            &mut self.u0,
            &mut self.v0,
            &mut self.dens,
            &mut self.dens0,
            &mut self.curl,
        ] {
            buf.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        // External velocity forces (into the velocity source arrays u0/v0).
        self.apply_buoyancy();
        if self.vorticity > 0.0 {
            self.apply_vorticity();
        }

        self.vel_step(dt);
        self.dens_step(dt);
        self.dissipate(dt);

        // Sources are per-frame: clear for the next one.
        self.u0.iter_mut().for_each(|x| *x = 0.0);
        self.v0.iter_mut().for_each(|x| *x = 0.0);
        self.dens0.iter_mut().for_each(|x| *x = 0.0);
    }

    fn apply_buoyancy(&mut self) {
        let n = self.n;
        for j in 1..=n {
            for i in 1..=n {
                let k = ix(i, j, n);
                self.v0[k] += self.buoyancy * self.dens[k];
            }
        }
    }

    fn apply_vorticity(&mut self) {
        let n = self.n;
        // curl (z-component of the 2D velocity curl).
        for j in 1..=n {
            for i in 1..=n {
                self.curl[ix(i, j, n)] = 0.5
                    * ((self.v[ix(i + 1, j, n)] - self.v[ix(i - 1, j, n)])
                        - (self.u[ix(i, j + 1, n)] - self.u[ix(i, j - 1, n)]));
            }
        }
        // Confinement force = eps * (N x w), N = grad|w| / |grad|w||.
        for j in 1..=n {
            for i in 1..=n {
                let dwdx =
                    0.5 * (self.curl[ix(i + 1, j, n)].abs() - self.curl[ix(i - 1, j, n)].abs());
                let dwdy =
                    0.5 * (self.curl[ix(i, j + 1, n)].abs() - self.curl[ix(i, j - 1, n)].abs());
                let len = (dwdx * dwdx + dwdy * dwdy).sqrt() + 1e-5;
                let nx = dwdx / len;
                let ny = dwdy / len;
                let w = self.curl[ix(i, j, n)];
                let k = ix(i, j, n);
                self.u0[k] += self.vorticity * ny * w;
                self.v0[k] += self.vorticity * -nx * w;
            }
        }
    }

    fn dissipate(&mut self, dt: f32) {
        if self.dissipation > 0.0 {
            let f = 1.0 / (1.0 + dt * self.dissipation);
            self.dens.iter_mut().for_each(|d| *d *= f);
        }
    }

    fn vel_step(&mut self, dt: f32) {
        let n = self.n;
        add_source(&mut self.u, &self.u0, dt);
        add_source(&mut self.v, &self.v0, dt);

        std::mem::swap(&mut self.u0, &mut self.u);
        diffuse(n, 1, &mut self.u, &self.u0, self.visc, dt);
        std::mem::swap(&mut self.v0, &mut self.v);
        diffuse(n, 2, &mut self.v, &self.v0, self.visc, dt);
        project(n, &mut self.u, &mut self.v, &mut self.u0, &mut self.v0);

        std::mem::swap(&mut self.u0, &mut self.u);
        std::mem::swap(&mut self.v0, &mut self.v);
        advect(n, 1, &mut self.u, &self.u0, &self.u0, &self.v0, dt);
        advect(n, 2, &mut self.v, &self.v0, &self.u0, &self.v0, dt);
        project(n, &mut self.u, &mut self.v, &mut self.u0, &mut self.v0);
    }

    fn dens_step(&mut self, dt: f32) {
        let n = self.n;
        add_source(&mut self.dens, &self.dens0, dt);
        std::mem::swap(&mut self.dens0, &mut self.dens);
        diffuse(n, 0, &mut self.dens, &self.dens0, self.diff, dt);
        std::mem::swap(&mut self.dens0, &mut self.dens);
        advect(n, 0, &mut self.dens, &self.dens0, &self.u, &self.v, dt);
    }
}

fn add_source(x: &mut [f32], s: &[f32], dt: f32) {
    for (xi, si) in x.iter_mut().zip(s) {
        *xi += dt * si;
    }
}

/// Boundary conditions. `b`: 0 = scalar (density), 1 = u-velocity, 2 = v-velocity
/// (velocity components are reflected at the matching wall to keep flow inside).
fn set_bnd(n: usize, b: u32, x: &mut [f32]) {
    for i in 1..=n {
        x[ix(0, i, n)] = if b == 1 { -x[ix(1, i, n)] } else { x[ix(1, i, n)] };
        x[ix(n + 1, i, n)] = if b == 1 { -x[ix(n, i, n)] } else { x[ix(n, i, n)] };
        x[ix(i, 0, n)] = if b == 2 { -x[ix(i, 1, n)] } else { x[ix(i, 1, n)] };
        x[ix(i, n + 1, n)] = if b == 2 { -x[ix(i, n, n)] } else { x[ix(i, n, n)] };
    }
    x[ix(0, 0, n)] = 0.5 * (x[ix(1, 0, n)] + x[ix(0, 1, n)]);
    x[ix(0, n + 1, n)] = 0.5 * (x[ix(1, n + 1, n)] + x[ix(0, n, n)]);
    x[ix(n + 1, 0, n)] = 0.5 * (x[ix(n, 0, n)] + x[ix(n + 1, 1, n)]);
    x[ix(n + 1, n + 1, n)] = 0.5 * (x[ix(n, n + 1, n)] + x[ix(n + 1, n, n)]);
}

/// Gauss-Seidel relaxation for the implicit diffusion and pressure solves.
fn lin_solve(n: usize, b: u32, x: &mut [f32], x0: &[f32], a: f32, c: f32) {
    for _ in 0..20 {
        for j in 1..=n {
            for i in 1..=n {
                x[ix(i, j, n)] = (x0[ix(i, j, n)]
                    + a * (x[ix(i - 1, j, n)]
                        + x[ix(i + 1, j, n)]
                        + x[ix(i, j - 1, n)]
                        + x[ix(i, j + 1, n)]))
                    / c;
            }
        }
        set_bnd(n, b, x);
    }
}

fn diffuse(n: usize, b: u32, x: &mut [f32], x0: &[f32], diff: f32, dt: f32) {
    let a = dt * diff * n as f32 * n as f32;
    lin_solve(n, b, x, x0, a, 1.0 + 4.0 * a);
}

/// Semi-Lagrangian advection: trace each cell back along the velocity field and
/// bilinearly sample the previous field there.
fn advect(n: usize, b: u32, d: &mut [f32], d0: &[f32], u: &[f32], v: &[f32], dt: f32) {
    let dt0 = dt * n as f32;
    let nf = n as f32;
    for j in 1..=n {
        for i in 1..=n {
            let mut x = i as f32 - dt0 * u[ix(i, j, n)];
            let mut y = j as f32 - dt0 * v[ix(i, j, n)];
            x = x.clamp(0.5, nf + 0.5);
            y = y.clamp(0.5, nf + 0.5);
            let i0 = x as usize;
            let i1 = i0 + 1;
            let j0 = y as usize;
            let j1 = j0 + 1;
            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;
            d[ix(i, j, n)] = s0 * (t0 * d0[ix(i0, j0, n)] + t1 * d0[ix(i0, j1, n)])
                + s1 * (t0 * d0[ix(i1, j0, n)] + t1 * d0[ix(i1, j1, n)]);
        }
    }
    set_bnd(n, b, d);
}

/// Hodge projection: subtract the gradient of the pressure (solved from the
/// velocity divergence) so the velocity becomes divergence-free. `p` and `div`
/// are scratch buffers.
fn project(n: usize, u: &mut [f32], v: &mut [f32], p: &mut [f32], div: &mut [f32]) {
    let h = 1.0 / n as f32;
    for j in 1..=n {
        for i in 1..=n {
            div[ix(i, j, n)] = -0.5
                * h
                * (u[ix(i + 1, j, n)] - u[ix(i - 1, j, n)] + v[ix(i, j + 1, n)]
                    - v[ix(i, j - 1, n)]);
            p[ix(i, j, n)] = 0.0;
        }
    }
    set_bnd(n, 0, div);
    set_bnd(n, 0, p);
    lin_solve(n, 0, p, div, 1.0, 4.0);
    for j in 1..=n {
        for i in 1..=n {
            u[ix(i, j, n)] -= 0.5 * (p[ix(i + 1, j, n)] - p[ix(i - 1, j, n)]) / h;
            v[ix(i, j, n)] -= 0.5 * (p[ix(i, j + 1, n)] - p[ix(i, j - 1, n)]) / h;
        }
    }
    set_bnd(n, 1, u);
    set_bnd(n, 2, v);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inject a puff at the bottom and check it stays finite/bounded and rises
    /// (centre of mass moves up) under buoyancy — the headless guard for smoke.
    #[test]
    fn smoke_is_stable_and_rises() {
        let n = 64;
        let mut s = Smoke::new(n);
        let cx = n / 2;
        let src_j = 6;

        for step in 0..240 {
            if step < 80 {
                for dj in 0..3 {
                    for di in -2i32..=2 {
                        let i = (cx as i32 + di) as usize;
                        s.add_density(i, src_j + dj, 8.0);
                        s.add_velocity(i, src_j + dj, 0.0, 8.0);
                    }
                }
            }
            s.step(1.0 / 60.0);
        }

        let mut total = 0.0f32;
        let mut weighted_j = 0.0f32;
        let mut max_d = 0.0f32;
        for j in 1..=n {
            for i in 1..=n {
                let d = s.density_at(i, j);
                assert!(d.is_finite(), "density NaN/inf at {i},{j}");
                total += d;
                weighted_j += d * j as f32;
                max_d = max_d.max(d);
            }
        }
        assert!(total > 1.0, "all smoke vanished (total {total})");
        assert!(max_d < 1e4, "density blew up (max {max_d})");
        let com_j = weighted_j / total;
        assert!(
            com_j > (src_j + 3) as f32,
            "smoke did not rise (com_j {com_j}, source {src_j})"
        );
        println!("smoke ok: total={total:.0} max={max_d:.1} com_j={com_j:.1}");
    }
}
