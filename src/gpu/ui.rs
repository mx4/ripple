//! egui overlay: a live tuning/profiling panel drawn on top of the simulation.
//! Wraps the egui ↔ winit ↔ wgpu plumbing so the app just feeds it events and a
//! closure that builds the UI.

use crate::gpu::Gpu;
use winit::window::Window;

pub struct EguiOverlay {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl EguiOverlay {
    pub fn new(gpu: &Gpu, window: &Window) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.format,
            egui_wgpu::RendererOptions::default(),
        );
        Self {
            ctx,
            state,
            renderer,
        }
    }

    /// Feed a window event to egui. Returns true if egui consumed it (so the app
    /// can skip its own handling — e.g. don't shake the fluid while dragging a
    /// slider).
    pub fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Build and draw the UI over `view` (loads the existing contents, so the
    /// sim must have rendered first). `build` adds the widgets.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        window: &Window,
        view: &wgpu::TextureView,
        mut build: impl FnMut(&egui::Context),
    ) {
        let raw = self.state.take_egui_input(window);
        let output = self.ctx.run_ui(raw, |ui| build(ui.ctx()));
        self.state
            .handle_platform_output(window, output.platform_output);

        let ppp = output.pixels_per_point;
        let jobs = self.ctx.tessellate(output.shapes, ppp);
        for (id, delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(&gpu.device, &gpu.queue, *id, delta);
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: ppp,
        };

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui") });
        let user_bufs = self
            .renderer
            .update_buffers(&gpu.device, &gpu.queue, &mut enc, &jobs, &screen);
        {
            let mut rpass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // keep the sim's render
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.renderer.render(&mut rpass, &jobs, &screen);
        }
        gpu.queue
            .submit(user_bufs.into_iter().chain([enc.finish()]));

        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
