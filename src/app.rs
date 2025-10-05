use winit::application::ApplicationHandler;

use crate::renderer::vulkan::VulkanRenderer;

pub struct App {
    vk: VulkanRenderer,
}

impl App {
    pub fn new() -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        Ok(Self {
            vk: VulkanRenderer::new(entry),
        })
    }

    pub unsafe fn intialize(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        unsafe {
            self.vk.initialize(event_loop)?;
        }

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        unsafe {
            if let Err(e) = self.intialize(event_loop) {
                tracing::error!("Failed to initialize application: {}", e);
            }
        }

        let ui = self.vk.update_ui();

        if let Err(e) = self.vk.draw_frame(ui) {
            tracing::error!("Error: {e}")
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(event) = self.vk.handle_egui_event(event) else {
            // egui handled the event
            return;
        };

        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                let ui = self.vk.update_ui();

                if let Err(e) = self.vk.draw_frame(ui) {
                    tracing::error!("Error: {e}")
                }
            }
            _ => {}
        }
    }
}
