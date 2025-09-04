use ash::vk;
use winit::event_loop::EventLoop;

pub mod app;

#[repr(C)]
#[derive(Clone, Debug, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
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
    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
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
                offset: std::mem::size_of::<[f32; 3]>() as u32, // After position field
            },
        ]
    }
}

// Triangle vertices
const VERTICES: [Vertex; 3] = [
    Vertex {
        pos: [0.0, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    }, // Top vertex, red
    Vertex {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    }, // Bottom right, green
    Vertex {
        pos: [-0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    }, // Bottom left, blue
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = app::App::new()?;

    event_loop.run_app(&mut app)?;

    app.cleanup();

    Ok(())
}
