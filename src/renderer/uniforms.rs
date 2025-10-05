pub struct UniformBufferObject {
    /// Camera view transform
    pub view: glam::Mat4,
    /// Perspective projection
    pub projection: glam::Mat4,
}

pub struct PushConstants {
    /// Model transform (object position/rotation)
    pub model: glam::Mat4,
}
