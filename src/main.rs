use winit::event_loop::EventLoop;

pub mod app;
pub mod components;
pub mod renderer;
pub mod resources;
pub mod systems;

pub mod shapes;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = app::App::new()?;

    event_loop.run_app(&mut app)?;

    Ok(())
}
