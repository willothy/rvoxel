// use glam::{Mat4, Vec3};
//
// pub struct Camera {
//     position: Vec3,
//     rotation: Vec3, // Pitch, yaw, roll in radians
// }
//
// impl Camera {
//     pub fn new() -> Self {
//         Camera {
//             position: Vec3::new(0.1, 0.1, 0.3), // Back away from cube
//             rotation: Vec3::ZERO,
//         }
//     }
//
//     pub fn get_view_matrix(&self) -> Mat4 {
//         // Simple look-at style view
//         Mat4::look_at_rh(
//             self.position, // Camera position
//             Vec3::ZERO,    // Look at origin (where cube is)
//             Vec3::Y,       // Up direction
//         )
//     }
//
//     pub fn get_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
//         // Perspective projection
//         Mat4::perspective_rh(
//             70.0_f32.to_radians(), // FOV (70 degrees)
//             aspect_ratio,          // Aspect ratio
//             0.01,                  // Near plane
//             100.0,                 // Far plane
//         )
//     }
// }
