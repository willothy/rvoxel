use std::sync::Arc;

use bevy_ecs::{intern::Interned, prelude::*, query::QuerySingleError, schedule::ScheduleLabel};
use glam::Vec3;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    keyboard::{KeyCode, PhysicalKey},
};

use crate::{
    components::{
        camera::{Camera, DebugCameraController},
        mesh::Mesh,
        transform::Transform,
    },
    debug::DebugWindow,
    renderer::vulkan::{context::VkContext, VulkanRenderer},
    resources::{input::InputState, time::Time, window_handle::WindowHandle},
};

pub struct App {
    world: World,
    schedule: Interned<dyn ScheduleLabel>,

    debug_window: DebugWindow,
}

impl App {
    pub fn new() -> anyhow::Result<Self> {
        let mut world = World::new();

        world.insert_resource(crate::resources::time::Time::default());
        world.insert_resource(crate::resources::input::InputState::default());

        let vk_ctx = Arc::new(VkContext::new()?);
        let vk = VulkanRenderer::new(vk_ctx.clone());
        world.insert_resource(vk);

        let mut schedule = Schedule::default();
        schedule.add_systems((
            crate::systems::debug_camera::debug_camera_system,
            // crate::resources::input::reset_input_system,
        ));

        world.spawn((
            Transform::from_xyz(0.0, 0.0, 0.3),
            Camera::default(),
            DebugCameraController::default(),
        ));

        world.spawn((
            Transform::default(),
            Mesh {
                vertices: crate::shapes::CUBE_VERTICES.to_vec(),
                indices: crate::shapes::CUBE_INDICES.to_vec(),
            },
        ));

        let label = schedule.label();
        world.add_schedule(schedule);

        Ok(Self {
            world,
            schedule: label,
            debug_window: DebugWindow::new(vk_ctx),
        })
    }

    pub fn renderer(&self) -> VulkanRenderer {
        self.world
            .get_resource::<VulkanRenderer>()
            .expect("VulkanRenderer resource not found")
            .clone()
    }

    pub unsafe fn intialize(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        let renderer = self.renderer();
        unsafe {
            renderer.initialize(event_loop)?;
            self.debug_window.initialize(event_loop)?;
        }

        self.world
            .insert_resource(crate::resources::window_handle::WindowHandle {
                window: self.renderer().window(),
            });

        Ok(())
    }

    pub fn camera_and_transform(&mut self) -> Result<(&Camera, &Transform), QuerySingleError> {
        let mut query = self.world.query::<(&Camera, &Transform)>();
        query.single(&self.world)
    }

    pub fn meshes_with_transforms(&'_ mut self) -> Vec<(&Mesh, &Transform)> {
        let mut query = self.world.query::<(&Mesh, &Transform)>();
        query.iter(&self.world).collect::<Vec<_>>()
    }

    fn render_debug_ui(&mut self) -> anyhow::Result<()> {
        let camera_transform = self.camera_and_transform().unwrap().1.clone();
        let renderer = self.renderer();

        unsafe {
            self.debug_window.render(|ctx| {
                egui::SidePanel::new(egui::panel::Side::Left, egui::Id::new("debug_ui_sidepanel"))
                    .show(ctx, |ui| {
                        ui.heading("Performance");
                        ui.label(format!("FPS: {:.1}", *renderer.debug().fps.read()));
                        ui.label(format!(
                            "Frame time: {:.3}ms",
                            *renderer.debug().frame_time.read() * 1000.0
                        ));

                        ui.separator();

                        ui.heading("Rendering");

                        if ui
                            .checkbox(&mut *renderer.debug().wireframe.write(), "Wireframe")
                            .changed()
                        {
                            tracing::info!(
                                "Wireframe mode set to {}",
                                *renderer.debug().wireframe.read()
                            );
                        }

                        ui.separator();

                        ui.heading("Camera");

                        if ui.button("Reset Camera").clicked() {
                            tracing::info!("Camera reset!");
                        }

                        // Calculate forward from the actual rotation quaternion
                        let forward = camera_transform.rotation * Vec3::NEG_Z; // Rotate local -Z by camera rotation

                        ui.small("Transform");

                        ui.label(format!(
                            "Position: ({:.2}, {:.2}, {:.2})",
                            camera_transform.position.x,
                            camera_transform.position.y,
                            camera_transform.position.z
                        ));
                        ui.label(format!(
                            "Rotation: ({:.2}, {:.2}, {:.2})",
                            forward.x, forward.y, forward.z
                        ));
                    });
            })?;
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

        let (camera, cam_transform) = match self.camera_and_transform() {
            Ok((camera, transform)) => (camera.clone(), transform.clone()),
            Err(e) => {
                tracing::error!("Failed to get camera and transform: {}", e);
                return;
            }
        };

        if let Err(e) = self
            .renderer()
            .draw_frame(&cam_transform.clone(), &camera.clone())
        {
            tracing::error!("Error: {e}")
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // We need to clean up the Vulkan resources before the winit window is destroyed,
        // because the Vulkan resources need the window handle.
        unsafe {
            self.debug_window.cleanup();
            self.renderer().cleanup_vulkan();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.world.run_schedule(self.schedule);

        // self.world.run_schedule(self.schedule);
        self.world.resource_mut::<InputState>().reset_frame();

        self.renderer().window().request_redraw();
        self.debug_window.window().request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            let mut input = self.world.resource_mut::<InputState>();
            if input.cursor_locked {
                input.mouse_delta = (delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if window_id == self.debug_window.window().id() {
            let res = self.debug_window.handle_event(&event);
            if res.repaint {
                self.debug_window.window().request_redraw();
            }
            if res.consumed {
                return;
            }
        }

        if let winit::event::WindowEvent::Resized(new_size) = event {
            if window_id == self.renderer().window().id() {
                if new_size.width > 0 && new_size.height > 0 {
                    tracing::debug!("Main window resized to {:?}", new_size);
                    unsafe {
                        if let Err(e) = self.renderer().recreate_swapchain() {
                            tracing::error!("Failed to recreate main window swapchain: {}", e);
                        }
                    }
                }
            }
        }

        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                if window_id == self.renderer().window().id() {
                    let (camera, cam_transform) = match self.camera_and_transform() {
                        Ok((camera, transform)) => (camera.clone(), transform.clone()),
                        Err(e) => {
                            tracing::error!("Failed to get camera and transform: {}", e);
                            return;
                        }
                    };

                    self.world.resource_mut::<Time>().update();

                    if let Err(e) = self.renderer().draw_frame(&cam_transform, &camera) {
                        tracing::error!("Error: {e}")
                    }
                } else if window_id == self.debug_window.window().id() {
                    if let Err(e) = self.render_debug_ui() {
                        tracing::error!("Debug UI render error: {e}");
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(keycode) = key_event.physical_key {
                    let mut input = self.world.resource_mut::<InputState>();

                    match key_event.state {
                        winit::event::ElementState::Pressed => {
                            input.keys_pressed.insert(keycode);

                            // Toggle cursor lock
                            if keycode == KeyCode::Tab {
                                toggle_cursor_lock(&mut self.world);
                            }
                        }
                        winit::event::ElementState::Released => {
                            input.keys_pressed.remove(&keycode);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    if state == winit::event::ElementState::Pressed {
                        lock_cursor(&mut self.world);
                    }
                }
            }
            _ => {}
        }
    }
}

fn lock_cursor(world: &mut World) {
    let WindowHandle { window } = world.resource::<WindowHandle>();

    window.set_cursor_visible(false);
    window
        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .ok();
    world.resource_mut::<InputState>().cursor_locked = true;
}

fn toggle_cursor_lock(world: &mut World) {
    let WindowHandle { window } = world.resource::<WindowHandle>().clone();
    let mut input = world.resource_mut::<InputState>();

    if input.cursor_locked {
        window.set_cursor_visible(true);
        window
            .set_cursor_grab(winit::window::CursorGrabMode::None)
            .ok();
        input.cursor_locked = false;
    } else {
        lock_cursor(world);
    }
}
