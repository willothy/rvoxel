use crate::components::camera::DebugCameraController;
use crate::components::transform::Transform;
use crate::resources::input::InputState;
use crate::resources::time::Time;
use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use winit::keyboard::KeyCode;

pub fn debug_camera_system(
    mut query: Query<(&mut Transform, &mut DebugCameraController)>,
    input: Res<InputState>,
    time: Res<Time>,
) {
    for (mut transform, mut controller) in query.iter_mut() {
        if input.cursor_locked {
            // Mouse look
            controller.yaw -= input.mouse_delta.0 * controller.mouse_sensitivity;
            controller.pitch += input.mouse_delta.1 * controller.mouse_sensitivity;

            // Clamp pitch
            controller.pitch = controller
                .pitch
                .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
        }

        // Update rotation quaternion from yaw/pitch
        let yaw_quat = Quat::from_axis_angle(Vec3::Y, controller.yaw);
        let pitch_quat = Quat::from_axis_angle(Vec3::X, controller.pitch);
        transform.rotation = yaw_quat * pitch_quat;

        // Calculate forward/right from the rotation quaternion (not from yaw/pitch)
        // This ensures movement matches where the camera is actually looking
        let forward = transform.rotation * Vec3::NEG_Z; // Local -Z becomes world forward
        let right = transform.rotation * Vec3::X; // Local +X becomes world right
        let up = Vec3::Y; // World up

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
        if input.is_key_pressed(KeyCode::KeyE) {
            transform.position -= up * speed;
        }
        if input.is_key_pressed(KeyCode::KeyQ) {
            transform.position += up * speed;
        }
    }
}
