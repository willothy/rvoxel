use ash::vk;
use glam::{Mat4, Vec3};
use winit::event_loop::EventLoop;

pub mod app;
pub mod renderer;

pub struct Camera {
    position: Vec3,
    rotation: Vec3, // Pitch, yaw, roll in radians
}

impl Camera {
    fn new() -> Self {
        Camera {
            position: Vec3::new(0.1, 0.1, 0.3), // Back away from cube
            rotation: Vec3::ZERO,
        }
    }

    fn get_view_matrix(&self) -> Mat4 {
        // Simple look-at style view
        Mat4::look_at_rh(
            self.position, // Camera position
            Vec3::ZERO,    // Look at origin (where cube is)
            Vec3::Y,       // Up direction
        )
    }

    fn get_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        // Perspective projection
        Mat4::perspective_rh(
            70.0_f32.to_radians(), // FOV (70 degrees)
            aspect_ratio,          // Aspect ratio
            0.01,                  // Near plane
            100.0,                 // Far plane
        )
    }
}

pub struct UniformBufferObject {
    /// Model transform (object position/rotation)
    model: glam::Mat4,
    /// Camera view transform
    view: glam::Mat4,
    /// Perspective projection
    projection: glam::Mat4,
}

#[repr(C)]
#[derive(Clone, Debug, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pos: glam::Vec3,
    color: glam::Vec3,
    normal: glam::Vec3,
}

impl Vertex {
    // Describes the overall vertex input
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0) // Binding index
            .stride(std::mem::size_of::<Vertex>() as u32) // Bytes between vertices
            .input_rate(vk::VertexInputRate::VERTEX) // Per-vertex data (not per-instance)
    }

    // Describes each field in the vertex
    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            // Position field
            vk::VertexInputAttributeDescription {
                binding: 0,                           // Same binding as above
                location: 0,                          // layout(location = 0) in shader
                format: vk::Format::R32G32B32_SFLOAT, // 3x 32-bit floats
                offset: 0,                            // Position is at start of struct
            },
            // Color field
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,                          // layout(location = 1) in shader
                format: vk::Format::R32G32B32_SFLOAT, // 3x 32-bit floats
                offset: std::mem::size_of::<glam::Vec3>() as u32, // After position field
            },
            // Normal field
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32B32_SFLOAT, // 3x 32-bit floats
                offset: (std::mem::size_of::<glam::Vec3>() * 2) as u32,
            },
        ]
    }
}

pub mod shapes {
    use glam::Vec3;

    use super::Vertex;

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = app::App::new()?;

    event_loop.run_app(&mut app)?;

    Ok(())
}
