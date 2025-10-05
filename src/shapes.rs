use glam::Vec3;

use crate::components::mesh::Vertex;

// A 5cm cube centered at origin
const CUBE_SIZE: f32 = 0.05; // 5cm in meters

pub const CUBE_VERTICES: [Vertex; 24] = [
    // Front face (facing +Z) - Red tint
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 0.5),
        normal: Vec3::Z,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 0.5),
        normal: Vec3::Z,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 0.5),
        normal: Vec3::Z,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 0.5),
        normal: Vec3::Z,
    },
    // Back face (facing -Z) - Green tint
    Vertex {
        pos: Vec3::new(CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 0.5),
        normal: Vec3::NEG_Z,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 0.5),
        normal: Vec3::NEG_Z,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 0.5),
        normal: Vec3::NEG_Z,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 0.5),
        normal: Vec3::NEG_Z,
    },
    // Top face (facing +Y) - Blue tint
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(0.5, 0.5, 1.0),
        normal: Vec3::Y,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(0.5, 0.5, 1.0),
        normal: Vec3::Y,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 0.5, 1.0),
        normal: Vec3::Y,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 0.5, 1.0),
        normal: Vec3::Y,
    },
    // Bottom face (facing -Y) - Yellow tint
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(1.0, 1.0, 0.5),
        normal: Vec3::NEG_Y,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(1.0, 1.0, 0.5),
        normal: Vec3::NEG_Y,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 1.0, 0.5),
        normal: Vec3::NEG_Y,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 1.0, 0.5),
        normal: Vec3::NEG_Y,
    },
    // Right face (facing +X) - Magenta tint
    Vertex {
        pos: Vec3::new(CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 1.0),
        normal: Vec3::X,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 1.0),
        normal: Vec3::X,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 1.0),
        normal: Vec3::X,
    },
    Vertex {
        pos: Vec3::new(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(1.0, 0.5, 1.0),
        normal: Vec3::X,
    },
    // Left face (facing -X) - Cyan tint
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 1.0),
        normal: Vec3::NEG_X,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 1.0),
        normal: Vec3::NEG_X,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 1.0),
        normal: Vec3::NEG_X,
    },
    Vertex {
        pos: Vec3::new(-CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE),
        color: Vec3::new(0.5, 1.0, 1.0),
        normal: Vec3::NEG_X,
    },
];

pub const CUBE_INDICES: [u16; 36] = [
    0, 1, 2, 2, 3, 0, // Front
    4, 5, 6, 6, 7, 4, // Back
    8, 9, 10, 10, 11, 8, // Top
    12, 13, 14, 14, 15, 12, // Bottom
    16, 17, 18, 18, 19, 16, // Right
    20, 21, 22, 22, 23, 20, // Left
];

pub const TRIANGLE_VERTICES: [Vertex; 3] = [
    Vertex {
        pos: glam::Vec3::new(0.0, -0.5, 0.0),
        color: glam::Vec3::new(1.0, 0.0, 0.0),
        normal: glam::Vec3::new(0.0, 0.0, 1.0),
    }, // Top vertex, red
    Vertex {
        pos: glam::Vec3::new(0.5, 0.5, 0.0),
        color: glam::Vec3::new(0.0, 1.0, 0.0),
        normal: glam::Vec3::new(0.0, 0.0, 1.0),
    }, // Bottom right, green
    Vertex {
        pos: glam::Vec3::new(-0.5, 0.5, 0.0),
        color: glam::Vec3::new(0.0, 0.0, 1.0),
        normal: glam::Vec3::new(0.0, 0.0, 1.0),
    }, // Bottom left, blue
];
