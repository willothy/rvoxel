use std::collections::HashMap;

use crate::world::{chunk::Chunk, coords::ChunkCoords};

pub struct VoxelWorld {
    pub chunks: HashMap<ChunkCoords, Chunk>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn get_chunk(&self, coords: ChunkCoords) -> Option<&Chunk> {
        self.chunks.get(&coords)
    }

    pub fn get_chunk_mut(&mut self, coords: ChunkCoords) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coords)
    }

    pub fn insert_chunk(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.position, chunk);
    }
}
