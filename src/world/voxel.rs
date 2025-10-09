#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Voxel {
    Air = 0,
    Stone,
    Dirt,
    Grass,
}

impl Voxel {
    pub fn is_solid(self) -> bool {
        match self {
            Voxel::Air => false,
            _ => true,
        }
    }

    pub fn color(self) -> glam::Vec3 {
        match self {
            Voxel::Air => glam::vec3(0.0, 0.0, 0.0),
            Voxel::Stone => glam::vec3(0.5, 0.5, 0.5),
            Voxel::Dirt => glam::vec3(0.59, 0.29, 0.0),
            Voxel::Grass => glam::vec3(0.0, 0.8, 0.0),
        }
    }
}
