use vulkan::VulkanApp;
use winit::application::ApplicationHandler;

pub mod shaders;
pub mod vulkan;

pub struct App {
    entry: ash::Entry,

    vk: VulkanApp,
}

impl App {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            entry: unsafe { ash::Entry::load()? },
            vk: VulkanApp::new_uninit(),
        })
    }

    pub fn cleanup(&mut self) {
        unsafe { self.vk.cleanup() };
    }

    pub fn intialize(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        if self.vk.is_initialized() {
            return Ok(());
        }

        self.vk.initialize(&self.entry, event_loop)?;

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Err(e) = self.intialize(event_loop) {
            eprintln!("Failed to initialize application: {}", e);
        }

        let ui = self.vk.update_ui();

        if let Err(e) = unsafe { self.vk.draw_frame(ui) } {
            eprintln!("Error: {e}")
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if !self.vk.is_initialized() {
            return;
        }

        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.vk.window.request_redraw();

                // let ui = self.vk.update_ui();
                //
                // if let Err(e) = unsafe { self.vk.draw_frame(ui) } {
                //     eprintln!("Error: {e}")
                // }
            }
            _ => {}
        }
    }
}
