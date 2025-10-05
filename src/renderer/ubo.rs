pub struct UniformBufferObject {
    /// Model transform (object position/rotation)
    pub model: glam::Mat4,
    /// Camera view transform
    pub view: glam::Mat4,
    /// Perspective projection
    pub projection: glam::Mat4,
}
