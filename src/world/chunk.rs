use crate::world::{
    coords::{ChunkCoords, LocalCoords, CHUNK_SIZE},
    voxel::Voxel,
};

pub const CHUNK_BYTES: usize = (crate::world::coords::CHUNK_SIZE as usize).pow(3);

pub struct Chunk {
    pub position: ChunkCoords,
    pub dirty: bool,
    voxels: Box<[Voxel; CHUNK_BYTES as usize]>,
}

impl Chunk {
    pub fn new(position: ChunkCoords) -> Self {
        Self {
            position,
            dirty: true,
            voxels: Box::new([Voxel::Air; CHUNK_BYTES as usize]),
        }
    }

    pub fn get(&self, coords: LocalCoords) -> Voxel {
        let index = (coords.x + coords.y * CHUNK_SIZE + coords.z * CHUNK_SIZE.pow(2)) as usize;
        self.voxels[index]
    }
}
