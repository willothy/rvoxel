use std::{cell::OnceCell, sync::Arc};

use egui::ViewportBuilder;
use winit::event_loop::ActiveEventLoop;

pub struct DebugWindow {
    ctx: egui::Context,
    state: OnceCell<egui_winit::State>,
    window: OnceCell<Arc<winit::window::Window>>,
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
        let viewport = ViewportBuilder::default()
            .with_visible(true)
            .with_title("RVoxel Debug");

        let window = Arc::new(egui_winit::create_window(&self.ctx, ev, &viewport)?);

        let state = egui_winit::State::new(
            self.ctx.clone(),
            self.ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );

        self.state
            .set(state)
            .map_err(|_| ())
            .expect("should only be initialized once");
        self.window
            .set(window)
            .map_err(|_| ())
            .expect("should only be initialized once");

        Ok(())
    }

    pub fn window(&self) -> &Arc<winit::window::Window> {
        self.window.get().expect("DebugWindow not initialized")
    }

    pub fn state(&self) -> &egui_winit::State {
        self.state.get().expect("DebugWindow not initialized")
    }

    pub fn state_mut(&mut self) -> &mut egui_winit::State {
        self.state.get_mut().expect("DebugWindow not initialized")
    }

    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    pub fn draw(&mut self, draw_fn: impl FnMut(&egui::Context)) {
        let win = Arc::clone(self.window());

        let input = self.state_mut().take_egui_input(win.as_ref());

        let output = self.ctx.run(input, draw_fn);

        self.state_mut()
            .handle_platform_output(win.as_ref(), output.platform_output);
    }

    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) -> egui_winit::EventResponse {
        let win = Arc::clone(&self.window());

        self.state_mut().on_window_event(win.as_ref(), event)
    }
}
