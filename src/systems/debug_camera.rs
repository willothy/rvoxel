use crate::components::camera::DebugCameraController;
use crate::components::transform::Transform;
use crate::components::*;
use crate::resources::input::InputState;
use crate::resources::time::Time;
use crate::resources::*;
use bevy_ecs::prelude::*;
use glam::Vec3;
use winit::keyboard::KeyCode;

pub fn debug_camera_system(
    mut query: Query<(&mut Transform, &mut DebugCameraController)>,
    input: Res<InputState>,
    time: Res<Time>,
) {
    for (mut transform, mut controller) in query.iter_mut() {
        // Mouse look
        if input.cursor_locked {
            controller.yaw += input.mouse_delta.0 * controller.mouse_sensitivity;
            controller.pitch -= input.mouse_delta.1 * controller.mouse_sensitivity;

            // Clamp pitch
            controller.pitch = controller
                .pitch
                .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
        }

        // Calculate forward and right vectors
        let forward = Vec3::new(
            controller.yaw.cos() * controller.pitch.cos(),
            controller.pitch.sin(),
            controller.yaw.sin() * controller.pitch.cos(),
        )
        .normalize();

        let right = forward.cross(Vec3::Y).normalize();

        // Movement
        let speed = controller.movement_speed * time.delta;

        if input.is_key_pressed(KeyCode::KeyW) {
            transform.position += forward * speed;
        }
        if input.is_key_pressed(KeyCode::KeyS) {
            transform.position -= forward * speed;
        }
        if input.is_key_pressed(KeyCode::KeyD) {
            transform.position += right * speed;
        }
        if input.is_key_pressed(KeyCode::KeyA) {
            transform.position -= right * speed;
        }
        if input.is_key_pressed(KeyCode::Space) {
            transform.position += Vec3::Y * speed;
        }
        if input.is_key_pressed(KeyCode::ShiftLeft) {
            transform.position -= Vec3::Y * speed;
        }
    }
}
