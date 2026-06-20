# fluid

A real-time 2D fluid playground in Rust — several fluid models in **one window**,
switchable live, built on **winit + wgpu** with an **egui** tuning panel.

Backends (press the number keys to switch):

1. **CPU SPH** — Smoothed-Particle Hydrodynamics liquid (particles, rayon-parallel).
2. **smoke** — Eulerian "Stable Fluids" gas (grid: advect + pressure projection,
   buoyancy, vorticity confinement).
3. **FLIP water** — hybrid particle+grid liquid with a free surface.
4. **GPU SPH** — the SPH solver running entirely in wgpu compute shaders, with
   dots / metaball / marching-squares rendering.
5. **GPU smoke** — the Eulerian solver running entirely in wgpu compute shaders
   (semi-Lagrangian advection + a ping-pong Jacobi pressure projection),
   rendered straight from the density buffer.

Each backend implements a `Simulation` trait, so the app just drives whichever is
active; rendering goes through a small shared wgpu toolkit (particle dots, a field
texture, the GPU-SPH renderers).

## Run

```sh
cargo run --release
```

(Use `--release` — debug builds of the solvers are slow.) Needs a GPU (Metal /
Vulkan / DX12).

## Controls

| Key | Action |
|-----|--------|
| 1–5 | Switch backend: CPU SPH / smoke / FLIP / GPU SPH / GPU smoke |
| ← / → | Shake the fluid sideways (liquids) |
| Space | Shake the fluid upward (liquids) |
| ↑ / ↓ | Gravity strength (liquids) |
| C | Toggle container shape, rect ⇄ circle (SPH) |
| M | Cycle render mode: dots / metaballs / MS lines / MS fill (GPU SPH) |
| S / V | Toggle bottom source / vorticity (smoke) |
| Mouse drag | Inject smoke / push water |
| R | Reset the active backend |
| Esc | Quit |

The egui panel (top-left) shows FPS and per-backend stats, with live sliders
(gravity), render-mode buttons, and reset.

## Layout

Solvers are pure math (no rendering deps), each unit-tested headlessly:

- `src/sim.rs` — CPU SPH solver (`Sim`) + container (`Bounds`); also the single
  source of truth for the tuned constants (`sph_constants`), reused by the GPU.
- `src/smoke.rs` — Eulerian smoke solver (`Smoke`).
- `src/flip.rs` — FLIP/PIC water solver (`FlipSim`).

The wgpu app and its building blocks live in `src/gpu/`:

- `context.rs` — shared `Gpu` (device/queue/surface).
- `backend.rs` — the `Simulation` trait + `Input` snapshot.
- `sph_backend.rs` — GPU SPH backend; `cpu_backends.rs` — CPU SPH / smoke / FLIP
  backends (step the CPU solver, upload to a renderer).
- `sim.rs` (`GpuSim`) — the GPU SPH compute solver (`sph.wgsl`).
- `smoke_gpu.rs` (`GpuSmoke` + backend) — the GPU Eulerian smoke solver
  (`smoke.wgsl`) with a ping-pong Jacobi pressure projection (the reusable grid
  primitive), drawn by `smoke_render.wgsl`.
- `particles.rs` / `field.rs` — generic particle-dots and field-texture renderers
  for the CPU backends; `render.rs` (+ `render.wgsl`, `metaball_*.wgsl`,
  `marching_squares*.wgsl`) — the GPU-SPH renderer (dots / metaballs / MS).
- `ui.rs` — the `EguiOverlay`.
- `src/main.rs` — the app: owns the `Gpu`, the active `Box<dyn Simulation>`, and
  the overlay; runs the winit event loop.

## Tests & benchmarks

```sh
cargo test --release                                   # headless stability guards
cargo test --release sweep   -- --ignored --nocapture  # SPH stiffness/dt sweep
cargo test --release bench   -- --ignored --nocapture  # CPU SPH throughput (rayon)
cargo test --release gpu_bench   -- --ignored --nocapture  # GPU throughput vs N
cargo test --release gpu_profile -- --ignored --nocapture  # GPU per-pass timing
```

Every solver has a headless test (it runs with no window and asserts the fluid
stays finite, in-bounds, and behaves — settles / rises / spreads). The GPU test
runs the compute shaders and reads positions back, matching the CPU's settling.

## Performance notes

- GPU SPH keeps all state resident in GPU buffers and renders from them (no
  per-frame readback); it scales to ~100k particles in real time (~5 M
  particle-steps/s at 1.4k → ~238 M at 86k on an Apple GPU).
- The GPU neighbour search uses a **fixed-capacity bucket grid** (atomic per-cell
  counters). A cell-sorted (CSR) grid was tried for coalesced reads but, at these
  particle counts, the prefix-sum scan plus extra passes cost more than the
  coalescing saved (~1.3–1.6× slower) — so the bucket grid was kept. `gpu_profile`
  shows the two neighbour passes dominate (~84% of a step).

## Tuning knobs

- SPH (`src/sim.rs`): `DEFAULT_STIFFNESS` (incompressibility vs timestep), `VISC`,
  `H`. Per-backend `SPH_*` consts in `src/gpu/{sph_backend,cpu_backends}.rs`.
- Smoke (`src/smoke.rs`): `buoyancy`, `dissipation`, `vorticity`, `diff`, `visc`.
- FLIP (`src/flip.rs` / `FlipSim::config()`): `gravity`, `flip_ratio`, `iters`.
- Metaball / marching-squares look: consts in `src/gpu/metaball_*.wgsl` /
  `marching_squares*.wgsl`.

## Possible next steps

- GPU FLIP — reuse the Jacobi grid projection (already built for GPU smoke); the
  remaining piece is particle↔grid transfer (P2G needs float atomics or a sort).
- A parallel-scan primitive (for a sorted GPU neighbour grid / FLIP compaction).
- Surface tension / two fluids; resizable GPU simulation domain.
- Per-pass timing surfaced live in the egui panel.
