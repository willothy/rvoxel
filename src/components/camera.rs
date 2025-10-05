use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec3};

// Camera component
#[derive(Component)]
pub struct Camera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: 70.0_f32.to_radians(),
            near: 0.01,
            far: 1000.0,
        }
    }
}

// Debug camera controller
#[derive(Component)]
pub struct DebugCameraController {
    pub yaw: f32,
    pub pitch: f32,
    pub movement_speed: f32,
    pub mouse_sensitivity: f32,
}

impl Default for DebugCameraController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            movement_speed: 2.0, // 2 meters per second
            mouse_sensitivity: 0.002,
        }
    }
}
