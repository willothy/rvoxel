use crate::world::{
    coords::{ChunkCoords, LocalCoords, CHUNK_SIZE},
    voxel::Voxel,
};

pub const CHUNK_BYTES: usize = (crate::world::coords::CHUNK_SIZE as usize).pow(3);

pub struct Chunk {
    pub position: ChunkCoords,
    voxels: Box<[Voxel; CHUNK_BYTES]>,
}

impl Chunk {
    pub fn new(position: ChunkCoords) -> Self {
        Self {
            position,
            voxels: Box::new([Voxel::Air; CHUNK_BYTES as usize]),
        }
    }

    pub fn data(&self) -> &[Voxel; CHUNK_BYTES] {
        &self.voxels
    }

    pub fn new_sphere(position: ChunkCoords, radius: f32) -> Self {
        let mut chunk = Self::new(position);
        let center = glam::Vec3::splat(CHUNK_SIZE as f32 / 2.0);
        let radius_sq = radius * radius;

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let pos = glam::vec3(x as f32, y as f32, z as f32);
                    if pos.distance_squared(center) <= radius_sq {
                        chunk.set(LocalCoords::new(x, y, z), Voxel::Stone);
                    }
                }
            }
        }

        chunk
    }

    fn index_of(&self, coords: LocalCoords) -> usize {
        (coords.x + coords.y * CHUNK_SIZE + coords.z * CHUNK_SIZE.pow(2)) as usize
    }

    pub fn set(&mut self, coords: LocalCoords, voxel: Voxel) {
        self.voxels[self.index_of(coords)] = voxel;
    }

    pub fn get(&self, coords: LocalCoords) -> Voxel {
        self.voxels[self.index_of(coords)]
    }
}
