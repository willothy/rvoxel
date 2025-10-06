use std::cell::OnceCell;
use std::sync::Arc;

use bevy_ecs::{intern::Interned, prelude::*, query::QuerySingleError, schedule::ScheduleLabel};
use egui::ViewportId;
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
    renderer::vulkan::{context::VkContext, VulkanRenderer},
    resources::{input::InputState, time::Time, window_handle::WindowHandle},
};

pub struct App {
    world: World,
    schedule: Interned<dyn ScheduleLabel>,

    egui_ctx: egui::Context,
    egui_state: OnceCell<egui_winit::State>,
    egui_window: OnceCell<winit::window::Window>,
}

impl App {
    pub fn new() -> anyhow::Result<Self> {
        let mut world = World::new();

        world.insert_resource(crate::resources::time::Time::default());
        world.insert_resource(crate::resources::input::InputState::default());

        let vk = VulkanRenderer::new(Arc::new(VkContext::new()?));
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

        let egui_ctx = egui::Context::default();

        Ok(Self {
            world,
            schedule: label,

            egui_ctx,
            egui_state: OnceCell::new(),
            egui_window: OnceCell::new(),
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
        }

        self.world
            .insert_resource(crate::resources::window_handle::WindowHandle {
                window: self.renderer().window(),
            });

        let egui_viewport = egui::ViewportBuilder::default()
            .with_visible(true)
            // .with_title("RVoxel Debug")
            // .with_resizable(true)
            // .with_inner_size(egui::Vec2::new(800., 600.))
            // .with_titlebar_buttons_shown(true)
            // .with_taskbar(true)
        ;

        let egui_window = egui_winit::create_window(&self.egui_ctx, event_loop, &egui_viewport)?;

        // let attrs = winit::window::Window::default_attributes()
        //     .with_content_protected(false)
        //     .with_title("rvoxel debug")
        //     .with_visible(true);
        //
        // let egui_window = event_loop.create_window(attrs)?;
        //
        // egui_window.set_visible(true);

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            self.egui_ctx.viewport_id(),
            &egui_window,
            None,
            None,
            None,
        );

        self.egui_state
            .set(egui_state)
            .map_err(|_| ())
            .expect("should only be initialized once");
        self.egui_window
            .set(egui_window)
            .map_err(|_| ())
            .expect("should only be initialized once");

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

    fn draw_ui(&mut self) {
        let input = self
            .egui_state
            .get_mut()
            .unwrap()
            .take_egui_input(&self.egui_window.get().unwrap());

        let camera_transform = self.camera_and_transform().unwrap().1.clone();

        let ui = self.egui_ctx.run(input, |ctx| {
            egui::SidePanel::new(egui::panel::Side::Left, egui::Id::new("debug_ui_sidepanel"))
                .show(ctx, |ui| {
                    let renderer = self.renderer();

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
        });

        self.egui_state
            .get_mut()
            .unwrap()
            .handle_platform_output(&self.egui_window.get().unwrap(), ui.platform_output);
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

        if let Err(e) = self.renderer().draw_frame(
            &cam_transform.clone(),
            &camera.clone(),
            &self.meshes_with_transforms(),
        ) {
            tracing::error!("Error: {e}")
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // We need to clean up the Vulkan resources before the winit window is destroyed,
        // because the Vulkan resources need the window handle.
        unsafe { self.renderer().cleanup_vulkan() };
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.world.run_schedule(self.schedule);

        // self.world.run_schedule(self.schedule);
        self.world.resource_mut::<InputState>().reset_frame();
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
        if window_id == self.egui_window.get().unwrap().id() {
            let res = self
                .egui_state
                .get_mut()
                .unwrap()
                .on_window_event(self.egui_window.get().as_ref().unwrap(), &event);
            if res.repaint {
                self.egui_window.get().unwrap().request_redraw();
            }
            if res.consumed {
                return;
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

                    if let Err(e) = self.renderer().draw_frame(
                        &cam_transform,
                        &camera,
                        &self.meshes_with_transforms(),
                    ) {
                        tracing::error!("Error: {e}")
                    }
                } else if window_id == self.egui_window.get().unwrap().id() {
                    self.draw_ui();
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
