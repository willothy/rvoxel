use bevy_ecs::prelude::*;

// Window handle - needed for Vulkan surface creation
#[derive(Resource, Clone)]
pub struct WindowHandle {
    pub window: std::sync::Arc<winit::window::Window>,
}
