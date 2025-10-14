use bytemuck::Zeroable;
use glam::UVec3;

use crate::world::voxel::Voxel;

pub struct OctreeNode {
    /// If this is a leaf: stores voxel material/type
    /// If this is a branch: unused
    data: u8,

    /// For branches: indices of the 8 children in the octree's node array
    /// For leaves: all 0 (no children)
    child_indices: [u32; 8],
}

impl OctreeNode {
    pub fn is_leaf(&self) -> bool {
        self.child_indices.iter().all(|&idx| idx == 0)
    }

    pub fn voxel(&self) -> Voxel {
        debug_assert!(self.is_leaf());
        unsafe { std::mem::transmute(self.data) }
    }
}

pub struct Octree {
    nodes: Vec<OctreeNode>,
    #[allow(unused)]
    root: Option<u32>,
    #[allow(unused)]
    max_depth: u8,
}

impl Octree {
    #[allow(unused)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        let total_nodes = self.nodes.len();
        let leaf_nodes = self.nodes.iter().filter(|node| node.is_leaf()).count();

        (total_nodes, leaf_nodes)
    }

    pub fn from_voxels(voxels: &[Voxel], size: u32) -> Self {
        let mut nodes = Vec::new();
        let root = Self::build_recursive(voxels, size, UVec3::zeroed(), size, &mut nodes, 0);

        Octree {
            nodes,
            root: root.map(|r| r as u32),
            max_depth: size.ilog2() as u8,
        }
    }

    #[inline(always)]
    pub fn sample_voxel(voxels: &[Voxel], size: u32, offset: UVec3) -> Voxel {
        let index = (offset.x + offset.y * size + offset.z * size * size) as usize;
        voxels[index]
    }

    pub fn is_region_uniform(
        voxels: &[Voxel],
        size: u32,
        offset: UVec3,
        region_size: u32,
        first_voxel: Voxel,
    ) -> bool {
        for x in 0..region_size {
            for y in 0..region_size {
                for z in 0..region_size {
                    let voxel = Self::sample_voxel(voxels, size, offset + UVec3::new(x, y, z));
                    if voxel != first_voxel {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn child_offset(index: usize) -> (bool, bool, bool) {
        ((index & 1) != 0, (index & 2) != 0, (index & 4) != 0)
    }

    fn build_recursive(
        voxels: &[Voxel],
        size: u32,
        offset: UVec3,
        region_size: u32,
        nodes: &mut Vec<OctreeNode>,
        depth: u8,
    ) -> Option<usize> {
        let first_voxel = Self::sample_voxel(voxels, size, offset);
        let is_uniform = Self::is_region_uniform(voxels, size, offset, region_size, first_voxel);

        if is_uniform && first_voxel == Voxel::Air {
            // Empty, don't create node
            return None;
        }

        if is_uniform {
            let node = OctreeNode {
                data: first_voxel as u8,
                child_indices: [0; 8],
            };

            nodes.push(node);

            return Some(nodes.len() - 1);
        }

        let half_size = region_size / 2;

        let node_index = nodes.len();
        nodes.push(OctreeNode {
            data: 0,
            child_indices: [0; 8],
        });

        for child_idx in 0..8 {
            let (x_offset, y_offset, z_offset) = Self::child_offset(child_idx);

            let child_offset = UVec3::new(
                offset.x + if x_offset { half_size } else { 0 },
                offset.y + if y_offset { half_size } else { 0 },
                offset.z + if z_offset { half_size } else { 0 },
            );

            if let Some(child_node_idx) =
                Self::build_recursive(voxels, size, child_offset, half_size, nodes, depth + 1)
            {
                nodes[node_index].child_indices[child_idx] = child_node_idx as u32;
            }
        }

        Some(node_index)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_octree_construction() {
        use crate::world::chunk::Chunk;
        use crate::world::coords::CHUNK_SIZE;
        use crate::world::octree::Octree;

        let chunk = Chunk::new_sphere(glam::IVec3::new(0, 0, 0).into(), 12.0);

        let octree = Octree::from_voxels(chunk.data(), CHUNK_SIZE);

        let (total_nodes, leaf_nodes) = octree.stats();
        println!("Total nodes: {}", total_nodes);
        println!("Leaf nodes: {}", leaf_nodes);
        assert!(false);

        assert!(total_nodes > 1);
        assert!(leaf_nodes > 1);
    }
}
