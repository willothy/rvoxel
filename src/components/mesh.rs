use ash::vk;
use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec3};

// Mesh component - holds the actual geometry
#[derive(Component)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

// Vertex structure (keep your existing one)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Vec3,
    pub normal: Vec3,
}

impl Vertex {
    // Describes the overall vertex input
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0) // Binding index
            .stride(std::mem::size_of::<Vertex>() as u32) // Bytes between vertices
            .input_rate(vk::VertexInputRate::VERTEX) // Per-vertex data (not per-instance)
    }

    // Describes each field in the vertex
    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
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
