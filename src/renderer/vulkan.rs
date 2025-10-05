use std::{
    ffi::CStr,
    ops::Not,
    sync::{atomic::AtomicUsize, Mutex, OnceLock},
};

use anyhow::Context;
use ash::{khr::surface, vk};
use egui::mutex::RwLock;
use glam::Mat4;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::renderer::{camera::Camera, ubo::UniformBufferObject, vertex::Vertex};

struct RendererInner {
    #[allow(unused)]
    entry: ash::Entry,

    /// The window that we are rendering to, from [`winit`].
    window: winit::window::Window,

    /// The Vulkan instance.
    ///
    /// TODO: figure out and explain what this really is
    instance: ash::Instance,

    /// Physical device represents a GPU in the system that we have
    /// selected.
    #[allow(unused)]
    physical_device: ash::vk::PhysicalDevice,

    /// The logical device is our connection to the physical device.
    ///
    /// You can think of it like a session between the application and the GPU driver.
    device: ash::Device,

    /// Queue for submitting graphics commands.
    graphics_queue: ash::vk::Queue,

    /// The platform-specific window or display that we are rendering to.
    surface: ash::vk::SurfaceKHR,
    surface_loader: surface::Instance,

    /// The swap chain is Vulkan's version of double buffering.
    ///
    /// Where in OpenGL you would just render to the back buffer and call swap_buffers(),
    /// in Vulkan you manage 2-3 images and explicitly manage rendering to them and presenting
    /// them.
    swapchain: vk::SwapchainKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    /// The swapchain images.
    #[allow(unused)]
    swapchain_images: Vec<vk::Image>,
    /// The format of the images in the swapchain.
    #[allow(unused)]
    swapchain_format: vk::Format,
    /// The extent (width and height) of the swapchain images.
    swapchain_extent: vk::Extent2D,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    swapchain_image_views: Vec<vk::ImageView>,

    command_pool: vk::CommandPool,
    /// Where draw commands are recorded before being submitted to the GPU.
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: AtomicUsize,

    vertex_buffer: vk::Buffer,
    /// Actual GPU memory allocation for the vertex buffer.
    vertex_buffer_memory: vk::DeviceMemory,

    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,

    vert_shader_module: vk::ShaderModule,
    frag_shader_module: vk::ShaderModule,

    render_pass: vk::RenderPass,
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    egui_ctx: egui::Context,
    egui_winit: RwLock<egui_winit::State>,
    egui_renderer: Mutex<Option<egui_ash_renderer::Renderer>>,
}

pub struct DebugState {
    fps: RwLock<f32>,
    frame_time: RwLock<f32>,
    wireframe: RwLock<bool>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    entry: ash::Entry,

    vk: OnceLock<RendererInner>,

    debug: DebugState,
    camera: Camera,
}

impl VulkanRenderer {
    pub fn new(entry: ash::Entry) -> Self {
        Self {
            entry,
            vk: OnceLock::new(),
            debug: DebugState {
                fps: RwLock::new(0.),
                frame_time: RwLock::new(0.),
                wireframe: RwLock::new(false),
            },
            camera: Camera::new(),
        }
    }

    pub fn handle_egui_event(
        &self,
        event: winit::event::WindowEvent,
    ) -> Option<winit::event::WindowEvent> {
        self.vk().egui_handle_event(event)
    }

    pub fn draw_frame(&mut self, ui: egui::FullOutput) -> anyhow::Result<()> {
        unsafe {
            self.vk
                .get_mut()
                .expect("not initialized")
                .draw_frame(ui, &self.debug, &self.camera)
        }
    }

    pub unsafe fn initialize(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        let renderer = RendererInner::new(&self.entry, event_loop)?;

        self.vk
            .set(renderer)
            .map_err(|_| ())
            .expect("should only initialize once");

        Ok(())
    }

    fn vk(&self) -> &RendererInner {
        self.vk.get().expect("VulkanRenderer not initialized")
    }

    pub fn update_ui(&mut self) -> egui::FullOutput {
        let raw_input = {
            let mut egui_winit = self.vk().egui_winit.write();

            egui_winit.take_egui_input(&self.vk().window)
        };

        self.vk().egui_ctx.run(raw_input, |ctx| {
            self.draw_debug_ui(ctx);
        })
    }

    fn draw_debug_ui(&self, ctx: &egui::Context) {
        // egui::TopBottomPanel::bottom(egui::Id::new("debug_ui"))
        egui::SidePanel::new(egui::panel::Side::Left, egui::Id::new("debug_ui")).show(ctx, |ui| {
            ui.heading("Performance");
            ui.label(format!("FPS: {:.1}", *self.debug.fps.read()));
            ui.label(format!(
                "Frame time: {:.3}ms",
                *self.debug.frame_time.read() * 1000.0
            ));

            ui.separator();

            ui.heading("Rendering");

            if ui
                .checkbox(&mut *self.debug.wireframe.write(), "Wireframe")
                .changed()
            {
                println!("Wireframe mode set to {}", *self.debug.wireframe.read());
            }

            if ui.button("Reset Camera").clicked() {
                // TODO: Reset camera when we add it
                println!("Camera reset!");
            }

            ui.separator();
        });
    }
}

impl Drop for RendererInner {
    fn drop(&mut self) {
        unsafe {
            self.cleanup();
        }
    }
}

impl RendererInner {
    pub fn egui_handle_event(
        &self,
        event: winit::event::WindowEvent,
    ) -> Option<winit::event::WindowEvent> {
        let res = self
            .egui_winit
            .write()
            .on_window_event(&self.window, &event);

        if res.repaint {
            self.window.request_redraw();
        }

        res.consumed.not().then_some(event)
    }

    unsafe fn draw_frame(
        &mut self,
        ui: egui::FullOutput,
        debug: &DebugState,
        camera: &Camera,
    ) -> anyhow::Result<()> {
        let start_time = std::time::Instant::now();

        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fences[current_frame]], true, u64::MAX)?;
        }

        // Get next image from swapchain
        let (image_index, _is_suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                // Signal this when ready
                self.image_available_semaphores[current_frame],
                // Don't use fence here
                vk::Fence::null(),
            )?
        };

        // Reset fence for next time (only after we know we're using this frame)
        unsafe {
            self.device
                .reset_fences(&[self.in_flight_fences[current_frame]])?
        };

        unsafe { self.update_uniform_buffer(camera) };

        unsafe {
            self.device
                .reset_command_buffer(self.command_buffers[current_frame], Default::default())?
        };

        // Step 4: Record commands
        self.record_command_buffer(self.command_buffers[current_frame], image_index, ui.clone())?;

        // Step 5: Submit commands to GPU
        unsafe { self.submit_commands()? };

        // Step 6: Present the result
        unsafe { self.present_image(image_index)? };

        self.egui_renderer
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .free_textures(&ui.textures_delta.free)?;

        // Step 7: Move to next frame
        self.current_frame
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |current| Some((current + 1) % MAX_FRAMES_IN_FLIGHT),
            )
            .expect("should not fail");

        // Update timing at the end
        let frame_time = start_time.elapsed().as_secs_f32();
        *debug.frame_time.write() = frame_time;
        *debug.fps.write() = if frame_time > 0.0 {
            1.0 / frame_time
        } else {
            0.0
        };

        Ok(())
    }

    unsafe fn update_uniform_buffer(&self, camera: &Camera) {
        // Calculate matrices
        let model = Mat4::IDENTITY; // No transformation for now (cube at origin)

        let aspect_ratio = self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32;
        let view = camera.get_view_matrix();
        let projection = camera.get_projection_matrix(aspect_ratio);

        let ubo = UniformBufferObject {
            model,
            view,
            projection,
        };

        let current_image = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        // Copy to GPU memory
        let data_ptr = unsafe {
            self.device
                .map_memory(
                    self.uniform_buffers_memory[current_image],
                    0,
                    std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map uniform buffer memory")
                as *mut UniformBufferObject
        };

        unsafe {
            std::ptr::copy_nonoverlapping(&ubo, data_ptr, 1);

            self.device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }

    pub unsafe fn cleanup(&self) {
        unsafe {
            // Wait for all GPU work to finish before destroying anything
            self.device.device_wait_idle().unwrap();

            // Destroy synchronization objects (per-frame objects)
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            // Destroy command pool (automatically frees command buffers)
            self.device.destroy_command_pool(self.command_pool, None);

            // Destroy vertex buffer and its memory
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            // Cleaning up descriptor pool automatically frees descriptor sets
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            // Destroy graphics pipeline and layout
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            // Destroy framebuffers (one per swapchain image)
            for &framebuffer in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            // Destroy image views (one per swapchain image)
            for &image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(image_view, None);
            }

            // Destroy render pass
            self.device.destroy_render_pass(self.render_pass, None);

            // Destroy shader modules
            self.device
                .destroy_shader_module(self.vert_shader_module, None);
            self.device
                .destroy_shader_module(self.frag_shader_module, None);

            // Destroy swapchain (note: images are destroyed automatically)
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            // Destroy surface
            self.surface_loader.destroy_surface(self.surface, None);

            // Destroy egui renderer and state
            drop(self.egui_renderer.lock().unwrap().take());

            // Destroy logical device
            self.device.destroy_device(None);

            // Destroy instance (last!)
            self.instance.destroy_instance(None);
        }
    }

    pub fn new(
        entry: &ash::Entry,
        ev: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<Self> {
        let attrs = winit::window::Window::default_attributes()
            .with_content_protected(false)
            .with_title("rvoxel")
            .with_visible(true);

        let window = ev.create_window(attrs)?;

        let egui_ctx = egui::Context::default();

        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );

        let instance = unsafe { Self::create_instance(&entry, &window)? };

        let physical_device = unsafe { Self::pick_physical_device(&instance)? };

        let (device, graphics_queue, queue_family) =
            unsafe { Self::create_logical_device(&instance, &physical_device)? };

        let (surface, surface_loader) = unsafe { Self::create_surface(entry, &instance, &window)? };

        let (swapchain, swapchain_loader, surface_format, extent) = unsafe {
            Self::create_swap_chain(
                &instance,
                &physical_device,
                &device,
                &surface,
                &surface_loader,
                &window,
            )?
        };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let command_pool = unsafe { Self::create_command_pool(&device, queue_family)? };

        let command_buffers = unsafe { Self::create_command_buffers(&device, &command_pool)? };

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            unsafe { Self::create_sync_objects(&device)? };

        let (vertex_buffer, vertex_buffer_memory) =
            Self::create_vertex_buffer(&device, &instance, physical_device)?;

        let (index_buffer, index_buffer_memory) =
            unsafe { Self::create_index_buffer(&device, &instance, physical_device)? };

        let descriptor_set_layout = unsafe { Self::create_descriptor_set_layout(&device) };
        let descriptor_pool = unsafe { Self::create_descriptor_pool(&device) };
        let (uniform_buffers, uniform_buffers_memory) =
            unsafe { Self::create_uniform_buffers(&device, &instance, physical_device)? };
        let descriptor_sets = unsafe {
            Self::create_descriptor_sets(
                &device,
                descriptor_pool,
                descriptor_set_layout,
                &uniform_buffers,
            )?
        };

        let vert_shader_module = unsafe {
            Self::create_shader_module(&device, &super::shaders::compile_vertex_shader()?)?
        };
        let frag_shader_module = unsafe {
            Self::create_shader_module(&device, &super::shaders::compile_fragment_shader()?)?
        };

        let render_pass = unsafe { Self::create_render_pass(&device, surface_format.format)? };

        let (graphics_pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            &device,
            extent,
            render_pass,
            vert_shader_module,
            frag_shader_module,
            descriptor_set_layout,
        )?;

        let (swapchain_image_views, swapchain_framebuffers) = unsafe {
            Self::create_framebuffers(
                &device,
                &swapchain_images,
                render_pass,
                extent,
                surface_format.format,
            )?
        };

        let egui_renderer = egui_ash_renderer::Renderer::with_default_allocator(
            &instance,
            physical_device.clone(),
            device.clone(),
            render_pass.clone(),
            egui_ash_renderer::Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
                srgb_framebuffer: true,
            },
        )?;

        Ok(Self {
            entry: entry.clone(),

            window,
            instance,
            physical_device,
            device,
            graphics_queue,

            surface,
            surface_loader,

            swapchain,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
            swapchain_images,
            swapchain_loader,
            swapchain_framebuffers,
            swapchain_image_views,

            command_pool,
            command_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,

            current_frame: 0.into(),

            vertex_buffer,
            vertex_buffer_memory,

            index_buffer,
            index_buffer_memory,

            uniform_buffers,
            uniform_buffers_memory,

            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,

            vert_shader_module,
            frag_shader_module,

            render_pass,
            graphics_pipeline,
            pipeline_layout,
            egui_ctx,
            egui_winit: RwLock::new(egui_winit),
            egui_renderer: Mutex::new(Some(egui_renderer)),
        })
    }

    unsafe fn submit_commands(&self) -> anyhow::Result<()> {
        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);
        let wait_semaphores = [self.image_available_semaphores[current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[current_frame]];
        let command_buffers = [self.command_buffers[current_frame]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores) // Don't start until image is available
            .wait_dst_stage_mask(&wait_stages) // Wait specifically before color output
            .command_buffers(&command_buffers) // The commands to execute
            .signal_semaphores(&signal_semaphores); // Signal when rendering is done

        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[current_frame], // Signal fence when GPU work is done
            )?
        };

        Ok(())
    }

    unsafe fn present_image(&self, image_index: u32) -> anyhow::Result<()> {
        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);
        let wait_semaphores = [self.render_finished_semaphores[current_frame]];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores) // Don't present until rendering is done
            .swapchains(&swapchains) // Which swapchain to present to
            .image_indices(&image_indices); // Which image in that swapchain

        unsafe {
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)?
        };

        Ok(())
    }

    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_idx: u32,
        ui: egui::FullOutput,
    ) -> anyhow::Result<()> {
        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        // Start recording
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };

        // Start render pass
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.2, 0.3, 0.3, 1.0], // Dark teal background
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_idx as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE, // We'll record commands directly (not secondary command buffers)
            )
        };

        // Bind graphics pipeline
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        // Bind vertex buffer
        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0];
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets)
        };

        unsafe {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[current_frame]],
                &[],
            );

            self.device.cmd_draw_indexed(
                command_buffer,
                crate::shapes::CUBE_INDICES.len() as u32,
                1,
                0,
                0,
                0,
            );
        }

        // UI
        // In record_command_buffer, after your triangle rendering but before cmd_end_render_pass:

        // Get egui render data
        let output = ui;
        let clipped_primitives = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);

        // Render egui
        if !clipped_primitives.is_empty() {
            // println!(
            //     "Rendering egui with {} clipped primitives",
            //     clipped_primitives.len()
            // );

            let mut renderer = self.egui_renderer.lock().unwrap();

            renderer.as_mut().unwrap().set_textures(
                self.graphics_queue,
                self.command_pool,
                &output.textures_delta.set,
            )?;

            if let Err(e) = renderer.as_mut().unwrap().cmd_draw(
                command_buffer,
                vk::Extent2D {
                    width: self.swapchain_extent.width,
                    height: self.swapchain_extent.height,
                },
                output.pixels_per_point,
                &clipped_primitives,
                // &output.textures_delta,
            ) {
                eprintln!("Failed to render egui: {}", e);
            }
        }

        // End render pass
        unsafe { self.device.cmd_end_render_pass(command_buffer) };

        // Finish recording
        unsafe { self.device.end_command_buffer(command_buffer)? };

        Ok(())
    }

    unsafe fn create_shader_module(
        device: &ash::Device,
        code: &[u32],
    ) -> anyhow::Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);

        unsafe {
            device
                .create_shader_module(&create_info, None)
                .context("Failed to create shader module")
        }
    }

    unsafe fn create_swap_chain(
        instance: &ash::Instance,
        physical_device: &ash::vk::PhysicalDevice,
        device: &ash::Device,
        surface: &ash::vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        window: &Window,
    ) -> anyhow::Result<(
        vk::SwapchainKHR,
        ash::khr::swapchain::Device,
        vk::SurfaceFormatKHR,
        vk::Extent2D,
    )> {
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(
                physical_device.clone(),
                surface.clone(),
            )?
        };
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device.clone(), surface.clone())?
        };
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(
                physical_device.clone(),
                surface.clone(),
            )?
        };

        // Pick image count (2-3 is good)
        let image_count =
            (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
                capabilities.max_image_count
            } else {
                u32::MAX
            });

        // Pick format (first one is usually fine)
        let surface_format = formats[0];

        // Pick present mode
        //
        // FIFO is always supported, but try MAILBOX if available
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX // Triple buffering
        } else {
            vk::PresentModeKHR::FIFO // VSync
        };

        let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);

        let extent = if capabilities.current_extent.width != u32::MAX {
            // Surface tells us exactly what size to use
            capabilities.current_extent
        } else {
            // We need to figure out the size from the window
            let window_size = window.inner_size();

            vk::Extent2D {
                width: window_size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.clone())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            // Always 1 unless doing VR
            .image_array_layers(1)
            // We'll draw to these
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            // Only graphics queue uses them
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            // No rotation
            .pre_transform(capabilities.current_transform)
            // No transparency
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            // Don't care about pixels behind other windows
            .clipped(true);

        // use ash::extensions::khr::swapchain;
        let swap_chain = unsafe { swapchain_device.create_swapchain(&info, None)? };

        Ok((swap_chain, swapchain_device, surface_format, extent))
    }

    unsafe fn create_framebuffers(
        device: &ash::Device,
        swapchain_images: &[vk::Image],
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::Format,
    ) -> anyhow::Result<(Vec<vk::ImageView>, Vec<vk::Framebuffer>)> {
        let image_views: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe {
                    device
                        .create_image_view(&view_info, None)
                        .context("Failed to create image view")
                }
            })
            .collect::<Result<_, _>>()?;

        let framebuffers: Vec<vk::Framebuffer> = image_views
            .iter()
            .map(|&image_view| {
                let attachments = [image_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&framebuffer_info, None)
                        .context("Failed to create framebuffer")
                }
            })
            .collect::<Result<_, _>>()?;

        Ok((image_views, framebuffers))
    }

    unsafe fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0) // binding = 0 in shader
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX); // Used in vertex shader

        let bindings = [ubo_layout_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .expect("Failed to create descriptor set layout")
        }
    }

    unsafe fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate descriptor sets")?
        };

        // Update descriptor sets to point to uniform buffers
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize);

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));

            unsafe { device.update_descriptor_sets(&[descriptor_write], &[]) };
        }

        println!("Descriptor sets created");

        Ok(descriptor_sets)
    }

    unsafe fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        }
    }

    unsafe fn create_uniform_buffers(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>)> {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;

        let mut buffers = Vec::new();
        let mut buffers_memory = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = unsafe {
                device
                    .create_buffer(&buffer_info, None)
                    .context("Failed to create uniform buffer")?
            };

            let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
            let memory_type_index = unsafe {
                Self::find_memory_type(
                    instance,
                    physical_device,
                    mem_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?
            };

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            let buffer_memory = unsafe {
                device
                    .allocate_memory(&alloc_info, None)
                    .context("Failed to allocate uniform buffer memory")?
            };

            unsafe {
                device
                    .bind_buffer_memory(buffer, buffer_memory, 0)
                    .context("Failed to bind uniform buffer memory")?
            };

            buffers.push(buffer);
            buffers_memory.push(buffer_memory);
        }

        println!("Uniform buffers created");

        Ok((buffers, buffers_memory))
    }

    unsafe fn create_index_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size =
            (std::mem::size_of::<u16>() * crate::shapes::CUBE_INDICES.len()) as vk::DeviceSize;

        // Create buffer
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER) // Index buffer usage
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .context("Failed to create index buffer")?
        };

        // Allocate memory
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = unsafe {
            Self::find_memory_type(
                instance,
                physical_device,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?
        };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let buffer_memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .context("Failed to allocate index buffer memory")?
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .context("Failed to bind index buffer memory")?
        };

        // Upload index data
        let data_ptr = unsafe {
            device
                .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .context("Failed to map index buffer memory")? as *mut u16
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                crate::shapes::CUBE_INDICES.as_ptr(),
                data_ptr,
                crate::shapes::CUBE_INDICES.len(),
            )
        };

        unsafe { device.unmap_memory(buffer_memory) };

        println!(
            "Index buffer created with {} indices",
            crate::shapes::CUBE_INDICES.len()
        );

        Ok((buffer, buffer_memory))
    }

    unsafe fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> anyhow::Result<(ash::vk::SurfaceKHR, surface::Instance)> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.into(),
                window.window_handle()?.into(),
                None,
            )?
        };

        let surface_loader = surface::Instance::new(entry, &instance);

        Ok((surface, surface_loader))
    }

    unsafe fn pick_physical_device(
        instance: &ash::Instance,
    ) -> anyhow::Result<ash::vk::PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        let mut discrete = None;
        let mut integrated = None;
        let mut first = None;

        for pd in &physical_devices {
            first = first.or_else(|| Some(pd));

            let props = unsafe { instance.get_physical_device_properties(pd.clone()) };
            if props.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU {
                discrete = discrete.or_else(|| Some(pd));
                continue;
            }
            if props.device_type == ash::vk::PhysicalDeviceType::INTEGRATED_GPU {
                integrated = integrated.or_else(|| Some(pd));
                continue;
            }
        }

        discrete
            .or(integrated)
            .or(first)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No suitable physical device found"))
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &ash::vk::PhysicalDevice,
    ) -> anyhow::Result<(ash::Device, ash::vk::Queue, u32)> {
        // TODO: combine this and pick_physical_device to actually find the device that
        // supports the most features we want.

        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device.clone())
        };

        // Find a graphics queue family
        let graphics_queue_family_index = queue_family_properties
            .iter()
            .position(|info| info.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32)
            .ok_or_else(|| anyhow::anyhow!("No suitable queue family found"))?;

        // Request queues from that graphics queue family
        let queue_create_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&[1.0]);

        // .fill_mode_non_solid(true)
        let features = ash::vk::PhysicalDeviceFeatures::default();

        let device_extension_names = [
            // enables swapchain
            ash::khr::swapchain::NAME.as_ptr(),
        ];

        let device_create_info = ash::vk::DeviceCreateInfo::default()
            .enabled_features(&features)
            .enabled_extension_names(&device_extension_names)
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        // Create the device
        //
        // A Device (different from PhysicalDevice) is a logical connection to the physical device,
        // like a session.
        let device =
            unsafe { instance.create_device(physical_device.clone(), &device_create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };

        Ok((device, graphics_queue, graphics_queue_family_index))
    }

    unsafe fn create_instance(
        entry: &ash::Entry,
        window: &Window,
    ) -> anyhow::Result<ash::Instance> {
        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(std::ffi::CStr::from_bytes_with_nul(b"rvoxel\0")?)
            .application_version(ash::vk::make_api_version(0, 1, 0, 0))
            .engine_name(std::ffi::CStr::from_bytes_with_nul(b"No Engine\0")?)
            .engine_version(ash::vk::make_api_version(0, 1, 0, 0))
            .api_version(ash::vk::API_VERSION_1_0);

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle()?.into())?.to_vec();

        extension_names.push(
            std::ffi::CStr::from_bytes_with_nul(b"VK_KHR_portability_enumeration\0")?.as_ptr(),
        );

        let mut flags = ash::vk::InstanceCreateFlags::default();

        flags |= ash::vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;

        let create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(flags);

        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    unsafe fn create_command_pool(
        device: &ash::Device,
        graphics_family_index: u32,
    ) -> anyhow::Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_family_index);

        Ok(unsafe { device.create_command_pool(&pool_info, None)? })
    }

    unsafe fn create_command_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> anyhow::Result<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool.clone())
            .level(vk::CommandBufferLevel::PRIMARY) // Can be submitted directly to queue
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

        unsafe { Ok(device.allocate_command_buffers(&alloc_info)?) }
    }

    unsafe fn create_sync_objects(
        device: &ash::Device,
    ) -> anyhow::Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // Start signaled so first frame doesn't wait
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
                render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
                in_flight_fences.push(device.create_fence(&fence_info, None)?);
            }
        }

        Ok((
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        ))
    }

    unsafe fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,                    // Bitmask of acceptable memory types
        properties: vk::MemoryPropertyFlags, // Required properties
    ) -> anyhow::Result<u32> {
        // Get all available memory types on this GPU
        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Check each memory type
        for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
            // Is this memory type suitable for our buffer? (type_filter is a bitmask)
            let type_suitable = (type_filter & (1 << i)) != 0;

            // Does this memory type have all the properties we need?
            let properties_suitable = memory_type.property_flags.contains(properties);

            if type_suitable && properties_suitable {
                return Ok(i as u32);
            }
        }

        return Err(anyhow::anyhow!(
            "Failed to find suitable memory type for buffer"
        ));
    }

    fn create_vertex_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size =
            (std::mem::size_of::<Vertex>() * crate::shapes::CUBE_VERTICES.len()) as vk::DeviceSize;

        // Create the buffer object
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER) // This will hold vertex data
            .sharing_mode(vk::SharingMode::EXCLUSIVE); // Only graphics queue uses it

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // Find out memory requirements
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Find suitable memory type
        let memory_type_index = unsafe {
            Self::find_memory_type(
                instance,
                physical_device,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?
        };

        // Allocate the memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let buffer_memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .context("Failed to allocate vertex buffer memory")?
        };

        // Connect buffer to memory
        unsafe {
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .context("Failed to bind vertex buffer memory")?
        };

        //  Upload vertex data
        unsafe {
            let data_ptr = device
                .map_memory(
                    buffer_memory,
                    0,                           // Offset
                    buffer_size,                 // Size to map
                    vk::MemoryMapFlags::empty(), // No special flags
                )
                .expect("Failed to map vertex buffer memory")
                as *mut Vertex;

            // Copy our vertex data
            std::ptr::copy_nonoverlapping(
                crate::shapes::CUBE_VERTICES.as_ptr(),
                data_ptr,
                crate::shapes::CUBE_VERTICES.len(),
            );

            // Unmap when done
            device.unmap_memory(buffer_memory);
        }

        Ok((buffer, buffer_memory))
    }

    unsafe fn create_render_pass(
        device: &ash::Device,
        swapchain_format: vk::Format,
    ) -> anyhow::Result<vk::RenderPass> {
        // Describe the color attachment (swapchain image)
        let color_attachment = vk::AttachmentDescription::default()
            .format(swapchain_format) // Same format as swapchain
            .samples(vk::SampleCountFlags::TYPE_1) // No multisampling
            .load_op(vk::AttachmentLoadOp::CLEAR) // Clear at start
            .store_op(vk::AttachmentStoreOp::STORE) // Keep result for presentation
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE) // No stencil
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED) // Don't care about initial state
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR); // Ready for presentation

        // Reference to the attachment
        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0) // Index 0 in attachments array
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL); // Layout during rendering

        // Describe the subpass
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        // Create the render pass
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass));

        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .context("Failed to create render pass")
        }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
        vert_shader_module: vk::ShaderModule,
        frag_shader_module: vk::ShaderModule,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> anyhow::Result<(vk::Pipeline, vk::PipelineLayout)> {
        // 1. Shader stages - which shaders to use
        let main_function_name = CStr::from_bytes_with_nul(b"main\0")?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(main_function_name),
        ];

        // 2. Vertex input - how to read vertex data
        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        // 3. Input assembly - what kind of primitives to draw
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST) // Groups of 3 vertices = triangles
            .primitive_restart_enable(false); // Don't use special restart index

        // 4. Viewport and scissor - what part of screen to render to
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        // 5. Rasterization - how to convert triangles to pixels
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false) // Don't clamp depth
            .rasterizer_discard_enable(false) // Don't discard before rasterization
            .polygon_mode(vk::PolygonMode::FILL) // Fill triangles (not wireframe)
            .line_width(1.0) // Line width for wireframe
            .cull_mode(vk::CullModeFlags::BACK) // Cull back faces
            .front_face(vk::FrontFace::CLOCKWISE) // Clockwise = front face
            .depth_bias_enable(false); // No depth bias

        // 6. Multisampling - anti-aliasing (disabled for now)
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // 7. Color blending - how to combine colors
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA) // Write all color components
            .blend_enable(false) // No blending for now
            ;

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        // 8. Pipeline layout - uniforms and push constants (none for now)
        let pipeline_layout = unsafe {
            let set_layouts = [descriptor_set_layout];

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts) // No descriptor sets
                .push_constant_ranges(&[]); // No push constants

            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create pipeline layout")?
        };

        // 9. Create the graphics pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0); // Subpass index

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(), // No pipeline cache for now
                    &[pipeline_info],
                    None,
                )
                .map_err(|(_res, e)| e)
                .context("Failed to create graphics pipeline")?[0]
        };

        println!("Graphics pipeline created");

        Ok((graphics_pipeline, pipeline_layout))
    }
}
