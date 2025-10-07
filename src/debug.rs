use std::{cell::OnceCell, sync::OnceLock};

use ash::vk;
use egui::{ViewportBuilder, ViewportId};
use winit::event_loop::ActiveEventLoop;

// TODO: Implement
pub struct DebugWindow {
    ctx: egui::Context,
    state: OnceCell<egui_winit::State>,
    window: OnceCell<winit::window::Window>,
    // surface: vk::SurfaceKHR,
}

impl DebugWindow {
    pub fn new() -> Self {
        Self {
            ctx: egui::Context::default(),
            state: OnceCell::new(),
            window: OnceCell::new(),
        }
    }

    pub fn initialize(&mut self, ev: &ActiveEventLoop) -> anyhow::Result<()> {
        let attrs = winit::window::Window::default_attributes()
            .with_content_protected(false)
            .with_title("rvoxel")
            .with_visible(true);

        let win = ev.create_window(attrs)?;

        let state = egui_winit::State::new(
            self.ctx.clone(),
            self.ctx.viewport_id(),
            &win,
            None,
            None,
            None,
        );

        self.state
            .set(state)
            .map_err(|_| ())
            .expect("should only be initialized once");
        self.window
            .set(win)
            .map_err(|_| ())
            .expect("should only be initialized once");

        Ok(())
    }
}
