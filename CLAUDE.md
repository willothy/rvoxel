# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVoxel is a Rust-based voxel rendering engine built with Vulkan (via `ash`) and using Bevy ECS for entity-component-system architecture. The project creates a 3D rendering environment with camera controls and real-time debugging UI via egui.

## Build & Development Commands

### Building
```bash
cargo build
cargo build --release
```

### Running
```bash
cargo run
cargo run --release
```

### Features
- `debug` (default): Enables Vulkan validation layers and debug messenger

## Architecture

### Core Structure

The application follows a three-tier architecture:

1. **App Layer** (`src/app.rs`): Main application handler implementing `winit::ApplicationHandler`
   - Manages the Bevy ECS `World` and `Schedule`
   - Handles window events, input, and renders both main window and debug UI
   - Coordinates between ECS systems and the Vulkan renderer

2. **ECS Layer**: Using Bevy ECS for game object management
   - **Components** (`src/components/`): Camera, Mesh, Transform
   - **Resources** (`src/resources/`): InputState, Time, WindowHandle
   - **Systems** (`src/systems/`): debug_camera (WASD movement + mouse look)

3. **Vulkan Renderer** (`src/renderer/vulkan/`):
   - **VkContext** (`context.rs`): Shared Vulkan resources (instance, device, queues, surface/swapchain loaders)
   - **VulkanRenderer** (`mod.rs`): Rendering pipeline, framebuffers, command buffers, synchronization

### Key Design Patterns

- **Lazy Initialization**: VulkanRenderer uses `OnceLock` since window creation requires an active event loop
- **Shared Context**: VkContext is wrapped in `Arc` and shared between renderer components to avoid duplication
- **Double Buffering**: Uses `MAX_FRAMES_IN_FLIGHT = 2` for frame overlap
- **Instanced Rendering**: Mesh transforms passed via push constants, allowing multiple instances

### Vulkan Pipeline

The rendering flow in `src/renderer/vulkan/mod.rs`:
1. Wait for previous frame's fence
2. Acquire next swapchain image
3. Update uniform buffers (camera matrices)
4. Record command buffer with meshes
5. Submit commands to GPU
6. Present image to swapchain

Shaders are compiled at runtime using `shaderc` from GLSL sources in `shaders/`:
- `vertex.vert.glsl`: Applies view/projection from UBO, model matrix from push constants
- `fragment.frag.glsl`: Per-vertex color output

### Window Management

The app manages two windows:
- **Main Window**: Vulkan rendering surface (created by VulkanRenderer)
- **Debug Window**: egui UI showing FPS, frame time, wireframe toggle, camera position

Input is captured via `InputState` resource:
- Tab: Toggle cursor lock
- Left-click: Lock cursor
- WASD + mouse: Camera movement (when locked)

## Common Tasks

### Adding a New Component
1. Create module in `src/components/`
2. Define component struct
3. Export from `src/components/mod.rs`
4. Spawn entities with component in `App::new()`

### Adding a New System
1. Create function in `src/systems/`
2. Add to schedule in `App::new()` via `schedule.add_systems()`
3. Systems run every frame in `about_to_wait`

### Modifying Shaders
- Edit GLSL files in `shaders/` directory
- Shaders are embedded and compiled at runtime via `include_str!`
- No separate build step needed; just rerun the application

### Vulkan Resource Management
- All Vulkan resources must be cleaned up in reverse creation order
- See `RendererInner::cleanup()` for proper teardown sequence
- Cleanup happens in `App::exiting()` before window destruction
