use anyhow::Context;

pub const VERTEX_SHADER: &str = include_str!("../../shaders/vertex.vert.glsl");
pub const FRAGMENT_SHADER: &str = include_str!("../../shaders/fragment.frag.glsl");

fn compile_shader(
    source: &str,
    shader_kind: shaderc::ShaderKind,
    file_name: &str,
) -> anyhow::Result<Vec<u32>> {
    let compiler = shaderc::Compiler::new()
        .context("Failed to create shader compiler. Is shaderc properly installed?")?;
    let mut options = shaderc::CompileOptions::new()
        .context("Failed to create shader compile options. Is shaderc properly installed?")?;

    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let binary_result =
        compiler.compile_into_spirv(source, shader_kind, file_name, "main", Some(&options))?;

    if binary_result.get_num_warnings() > 0 {
        tracing::error!(
            "Shader compilation warnings: {}",
            binary_result.get_warning_messages()
        );
    }

    Ok(binary_result.as_binary().to_vec())
}

pub fn compile_vertex_shader() -> anyhow::Result<Vec<u32>> {
    compile_shader(
        VERTEX_SHADER,
        shaderc::ShaderKind::Vertex,
        "vertex.vert.glsl",
    )
}

pub fn compile_fragment_shader() -> anyhow::Result<Vec<u32>> {
    compile_shader(
        FRAGMENT_SHADER,
        shaderc::ShaderKind::Fragment,
        "fragment.frag.glsl",
    )
}
