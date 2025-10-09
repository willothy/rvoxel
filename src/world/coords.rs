use std::{hash::Hash, ops::Deref};

use glam::{IVec3, UVec3, Vec3};

/// Diameter of a chunk in voxels
pub const CHUNK_SIZE: u32 = 32;

/// Size of a voxel in meters
pub const VOXEL_SIZE: f32 = 0.05;

/// Chunk position in world space (each unit = one chunk)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldCoords(Vec3);

impl WorldCoords {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
}

impl Deref for WorldCoords {
    type Target = Vec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec3> for WorldCoords {
    fn from(v: Vec3) -> Self {
        Self(v)
    }
}

impl From<ChunkCoords> for WorldCoords {
    fn from(c: ChunkCoords) -> Self {
        WorldCoords(Vec3::new(
            c.x as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE + (CHUNK_SIZE as f32 * VOXEL_SIZE) / 2.0,
            c.y as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE + (CHUNK_SIZE as f32 * VOXEL_SIZE) / 2.0,
            c.z as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE + (CHUNK_SIZE as f32 * VOXEL_SIZE) / 2.0,
        ))
    }
}

/// Chunk position in world space (each unit = one chunk)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoords(IVec3);

impl ChunkCoords {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }
}

impl Deref for ChunkCoords {
    type Target = IVec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<IVec3> for ChunkCoords {
    fn from(v: IVec3) -> Self {
        Self(v)
    }
}

/// Voxel position in world space (each unit = one voxel)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelCoords(IVec3);

impl VoxelCoords {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }
}

impl Deref for VoxelCoords {
    type Target = IVec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<IVec3> for VoxelCoords {
    fn from(v: IVec3) -> Self {
        Self(v)
    }
}

/// Local voxel position within a chunk (0..CHUNK_SIZE)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalCoords(UVec3);

impl LocalCoords {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self(UVec3::new(x, y, z))
    }

    pub fn is_valid(&self) -> bool {
        self.x < CHUNK_SIZE && self.y < CHUNK_SIZE && self.z < CHUNK_SIZE
    }
}

impl Deref for LocalCoords {
    type Target = UVec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<UVec3> for LocalCoords {
    fn from(v: UVec3) -> Self {
        Self(v)
    }
}

/// Convert world position (meters) to voxel position
pub fn world_to_voxel(world_pos: WorldCoords) -> VoxelCoords {
    VoxelCoords(IVec3::new(
        (world_pos.x / VOXEL_SIZE).floor() as i32,
        (world_pos.y / VOXEL_SIZE).floor() as i32,
        (world_pos.z / VOXEL_SIZE).floor() as i32,
    ))
}

/// Convert voxel position to world position (meters, center of voxel)
pub fn voxel_to_world(voxel_pos: VoxelCoords) -> WorldCoords {
    WorldCoords(Vec3::new(
        voxel_pos.x as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5,
        voxel_pos.y as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5,
        voxel_pos.z as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5,
    ))
}

/// Convert voxel position to chunk position
pub fn voxel_to_chunk(voxel_pos: VoxelCoords) -> ChunkCoords {
    ChunkCoords::new(
        voxel_pos.x.div_euclid(CHUNK_SIZE as i32),
        voxel_pos.y.div_euclid(CHUNK_SIZE as i32),
        voxel_pos.z.div_euclid(CHUNK_SIZE as i32),
    )
}

/// Convert voxel position to local position within chunk
pub fn voxel_to_local(voxel_pos: VoxelCoords) -> LocalCoords {
    LocalCoords::new(
        voxel_pos.x.rem_euclid(CHUNK_SIZE as i32) as u32,
        voxel_pos.y.rem_euclid(CHUNK_SIZE as i32) as u32,
        voxel_pos.z.rem_euclid(CHUNK_SIZE as i32) as u32,
    )
}

/// Convert chunk position and local position to voxel position
pub fn chunk_local_to_voxel(chunk_pos: ChunkCoords, local_pos: LocalCoords) -> VoxelCoords {
    VoxelCoords::new(
        chunk_pos.x * CHUNK_SIZE as i32 + local_pos.x as i32,
        chunk_pos.y * CHUNK_SIZE as i32 + local_pos.y as i32,
        chunk_pos.z * CHUNK_SIZE as i32 + local_pos.z as i32,
    )
}

/// Split voxel position into chunk and local
pub fn split_voxel_pos(voxel_pos: VoxelCoords) -> (ChunkCoords, LocalCoords) {
    (voxel_to_chunk(voxel_pos), voxel_to_local(voxel_pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_to_chunk() {
        assert_eq!(
            voxel_to_chunk(IVec3::new(0, 0, 0).into()),
            IVec3::new(0, 0, 0).into()
        );
        assert_eq!(
            voxel_to_chunk(IVec3::new(31, 31, 31).into()),
            IVec3::new(0, 0, 0).into()
        );
        assert_eq!(
            voxel_to_chunk(IVec3::new(32, 32, 32).into()),
            IVec3::new(1, 1, 1).into()
        );
        assert_eq!(
            voxel_to_chunk(IVec3::new(-1, -1, -1).into()),
            IVec3::new(-1, -1, -1).into()
        );
    }

    #[test]
    fn test_voxel_to_local() {
        assert_eq!(
            voxel_to_local(IVec3::new(0, 0, 0).into()),
            UVec3::new(0, 0, 0).into()
        );
        assert_eq!(
            voxel_to_local(IVec3::new(31, 31, 31).into()),
            UVec3::new(31, 31, 31).into()
        );
        assert_eq!(
            voxel_to_local(IVec3::new(32, 32, 32).into()),
            UVec3::new(0, 0, 0).into()
        );
        assert_eq!(
            voxel_to_local(IVec3::new(-1, -1, -1).into()),
            UVec3::new(31, 31, 31).into()
        );
    }

    #[test]
    fn test_round_trip() {
        let voxel_pos = IVec3::new(100, -50, 200).into();
        let (chunk_pos, local_pos) = split_voxel_pos(voxel_pos);
        let reconstructed = chunk_local_to_voxel(chunk_pos, local_pos);
        assert_eq!(voxel_pos, reconstructed);
    }
}
