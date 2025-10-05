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
