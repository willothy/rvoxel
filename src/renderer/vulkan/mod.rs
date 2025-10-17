use std::{
    ffi::CStr,
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};

use anyhow::Context;
use ash::{khr::surface, vk};
use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use parking_lot::RwLock;

use crate::{
    components::{camera::Camera, mesh::Vertex, transform::Transform},
    renderer::uniforms::UniformBufferObject,
    world::{
        chunk::{Chunk, CHUNK_BYTES},
        coords::{ChunkCoords, CHUNK_SIZE},
        octree::{Octree, OctreeNode},
    },
};

pub mod context;

struct RendererInner {
    ctx: Arc<context::VkContext>,

    /// The window that we are rendering to, from [`winit`].
    window: Arc<winit::window::Window>,

    /// The platform-specific window or display that we are rendering to.
    surface: ash::vk::SurfaceKHR,

    /// The swap chain is Vulkan's version of double buffering.
    ///
    /// Where in OpenGL you would just render to the back buffer and call swap_buffers(),
    /// in Vulkan you manage 2-3 images and explicitly manage rendering to them and presenting
    /// them.
    swapchain: vk::SwapchainKHR,

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

    /// Per-frame image semaphores
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<RwLock<vk::Fence>>,
    current_frame: AtomicUsize,

    // NOTE: old
    // vertex_buffer: vk::Buffer,
    // /// Actual GPU memory allocation for the vertex buffer.
    // vertex_buffer_memory: vk::DeviceMemory,
    //
    // index_buffer: vk::Buffer,
    // index_buffer_memory: vk::DeviceMemory,
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

    voxel_image: vk::Image,
    voxel_image_memory: vk::DeviceMemory,
    voxel_image_view: vk::ImageView,
    voxel_sampler: vk::Sampler,

    voxel_buffer: vk::Buffer,
    voxel_buffer_memory: vk::DeviceMemory,

    // Stores the octree nodes after they have been submitted to the GPU.
    //
    // This is temporary since we will need to update this buffer when the world changes.
    octree_buffer: vk::Buffer,
    octree_buffer_memory: vk::DeviceMemory,

    // Shared mesh data
    cube_vertex_buffer: vk::Buffer,
    cube_vertex_buffer_memory: vk::DeviceMemory,
    cube_index_buffer: vk::Buffer,
    cube_index_buffer_memory: vk::DeviceMemory,
}

pub struct DebugState {
    pub fps: RwLock<f32>,
    pub frame_time: RwLock<f32>,
    pub wireframe: RwLock<bool>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Resource, Clone)]
pub struct VulkanRenderer {
    vk: Arc<context::VkContext>,

    inner: Arc<OnceLock<RendererInner>>,

    debug: Arc<DebugState>,
}

impl VulkanRenderer {
    pub fn new(vk: Arc<context::VkContext>) -> Self {
        Self {
            vk,
            inner: Arc::new(OnceLock::new()),
            debug: Arc::new(DebugState {
                fps: RwLock::new(0.),
                frame_time: RwLock::new(0.),
                wireframe: RwLock::new(false),
            }),
        }
    }

    pub fn debug(&self) -> &Arc<DebugState> {
        &self.debug
    }

    pub unsafe fn cleanup_vulkan(&self) {
        unsafe { self.inner.get().unwrap().cleanup() };
    }

    pub fn window(&self) -> Arc<winit::window::Window> {
        Arc::clone(&self.inner.get().unwrap().window)
    }

    // pub fn handle_egui_event(
    //     &self,
    //     event: winit::event::WindowEvent,
    // ) -> Option<winit::event::WindowEvent> {
    //     self.vk().egui_handle_event(event)
    // }
    //
    // pub fn handle_mouse_motion_diff(&self, pos: (f64, f64)) {
    //     let Some(cur) = self.vk().egui_ctx.pointer_interact_pos() else {
    //         return;
    //     };
    //
    //     let dx = cur.x as f64 - pos.0;
    //     let dy = cur.y as f64 - pos.1;
    //
    //     self.handle_egui_mouse_motion((dx, dy));
    // }

    // pub fn handle_egui_mouse_motion(&self, delta: (f64, f64)) {
    //     self.vk().egui_handle_mouse_motion(delta);
    // }

    pub fn draw_frame(&self, camera_transform: &Transform, camera: &Camera) -> anyhow::Result<()> {
        let vk = self.inner.get().unwrap();

        unsafe { vk.draw_frame(&self.debug, camera, camera_transform) }
    }

    pub unsafe fn initialize(
        &self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        let renderer = RendererInner::new(&self.vk, event_loop)?;

        self.inner
            .set(renderer)
            .map_err(|_| ())
            .expect("should only initialize once");

        Ok(())
    }

    pub unsafe fn recreate_swapchain(&self) -> anyhow::Result<()> {
        let vk = self.vk();
        unsafe { vk.recreate_swapchain(&self.vk) }
    }

    fn vk(&self) -> &RendererInner {
        self.inner.get().expect("VulkanRenderer not initialized")
    }
}

impl RendererInner {
    pub fn new(
        ctx: &Arc<context::VkContext>,
        ev: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<Self> {
        let attrs = winit::window::Window::default_attributes()
            .with_content_protected(false)
            .with_title("rvoxel")
            .with_visible(true);

        let window = ev.create_window(attrs)?;

        let surface = unsafe { Self::create_surface(&ctx.entry, &ctx.instance, &window)? };

        let (swapchain, swapchain_loader, surface_format, extent) = unsafe {
            Self::create_swap_chain(
                &ctx.instance,
                &ctx.physical_device,
                &ctx.device,
                &surface,
                &ctx.surface_loader,
                &window,
            )?
        };

        let actual_window_size = window.inner_size();
        tracing::debug!(
            "Main window: after swapchain creation - window inner_size={:?}, swapchain extent={:?}",
            actual_window_size,
            extent
        );
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let command_pool =
            unsafe { Self::create_command_pool(&ctx.device, ctx.graphics_queue_family_index)? };

        let command_buffers = unsafe { Self::create_command_buffers(&ctx.device, &command_pool)? };

        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = unsafe { Self::create_sync_objects(&ctx.device, swapchain_images.len())? };

        let (cube_vertex_buffer, cube_vertex_buffer_memory) =
            Self::create_vertex_buffer(&ctx.device, &ctx.instance, ctx.physical_device)?;

        let (cube_index_buffer, cube_index_buffer_memory) =
            unsafe { Self::create_index_buffer(&ctx.device, &ctx.instance, ctx.physical_device)? };

        let (voxel_image, voxel_image_memory, voxel_image_view, voxel_sampler) = unsafe {
            Self::create_voxel_texture(&ctx.device, &ctx.instance, &ctx.physical_device)?
        };
        let (voxel_buffer, voxel_buffer_memory) =
            unsafe { Self::create_voxel_buffer(&ctx.device, &ctx.instance, &ctx.physical_device)? };

        let descriptor_set_layout = unsafe { Self::create_descriptor_set_layout(&ctx.device) };
        let descriptor_pool = unsafe { Self::create_descriptor_pool(&ctx.device) };
        let (uniform_buffers, uniform_buffers_memory) = unsafe {
            Self::create_uniform_buffers(&ctx.device, &ctx.instance, ctx.physical_device)?
        };
        let descriptor_sets = unsafe {
            Self::create_descriptor_sets(
                &ctx.device,
                descriptor_pool,
                descriptor_set_layout,
                &uniform_buffers,
                voxel_image_view.clone(),
                voxel_sampler.clone(),
            )?
        };

        let vert_shader_module = unsafe {
            Self::create_shader_module(&ctx.device, &super::shaders::compile_vertex_shader()?)?
        };
        let frag_shader_module = unsafe {
            Self::create_shader_module(&ctx.device, &super::shaders::compile_fragment_shader()?)?
        };

        let render_pass = unsafe { Self::create_render_pass(&ctx.device, surface_format.format)? };

        let (graphics_pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            &ctx.device,
            extent,
            render_pass,
            vert_shader_module,
            frag_shader_module,
            descriptor_set_layout,
        )?;

        let (swapchain_image_views, swapchain_framebuffers) = unsafe {
            Self::create_framebuffers(
                &ctx.device,
                &swapchain_images,
                render_pass,
                extent,
                surface_format.format,
            )?
        };

        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool.clone())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let buffers = ctx.device.allocate_command_buffers(&alloc_info)?;

            if buffers.len() != 1 {
                return Err(anyhow::anyhow!(
                    "Expected to allocate exactly 1 command buffer, got {}",
                    buffers.len()
                ));
            }

            let buffer = buffers[0];

            let begin = vk::CommandBufferBeginInfo::default();

            ctx.device.begin_command_buffer(buffer, &begin)?;

            {
                let image_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(voxel_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .src_access_mask(vk::AccessFlags::NONE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

                ctx.device.cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: CHUNK_SIZE,
                    height: CHUNK_SIZE,
                    depth: CHUNK_SIZE,
                });

            ctx.device.cmd_copy_buffer_to_image(
                buffer,
                voxel_buffer,
                voxel_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            {
                let image_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(voxel_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);

                ctx.device.cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            ctx.device.end_command_buffer(buffer)?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&buffer));

            ctx.device
                .queue_submit(ctx.graphics_queue, &[submit_info], vk::Fence::null())?;

            ctx.device.queue_wait_idle(ctx.graphics_queue)?;

            ctx.device.free_command_buffers(command_pool, &[buffer]);
        }

        let (octree_buffer, octree_buffer_memory) =
            unsafe { Self::create_octree_buffers(&ctx, &command_pool, &Octree::new())? };

        Ok(Self {
            ctx: Arc::clone(ctx),

            window: Arc::new(window),

            surface,

            swapchain,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
            swapchain_images,
            swapchain_framebuffers,
            swapchain_image_views,

            command_pool,
            command_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,

            current_frame: 0.into(),

            // vertex_buffer,
            // vertex_buffer_memory,
            //
            // index_buffer,
            // index_buffer_memory,
            cube_vertex_buffer,
            cube_vertex_buffer_memory,

            cube_index_buffer,
            cube_index_buffer_memory,

            uniform_buffers,
            uniform_buffers_memory,

            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,

            vert_shader_module,
            frag_shader_module,

            voxel_image,
            voxel_image_memory,
            voxel_image_view,
            voxel_sampler,

            voxel_buffer,
            voxel_buffer_memory,

            octree_buffer,
            octree_buffer_memory,

            render_pass,
            graphics_pipeline,
            pipeline_layout,
        })
    }

    pub unsafe fn cleanup(&self) {
        unsafe {
            // Wait for all GPU work to finish before destroying anything
            self.ctx.device.device_wait_idle().unwrap();

            // Destroy synchronization objects (per-frame objects)
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.ctx
                    .device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.ctx
                    .device
                    .destroy_fence(self.in_flight_fences[i], None);
            }

            // Destroy per-swapchain-image semaphores
            for i in 0..self.swapchain_images.len() {
                self.ctx
                    .device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
            }

            // Destroy command pool (automatically frees command buffers)
            self.ctx
                .device
                .destroy_command_pool(self.command_pool, None);

            // Destroy cube shared buffers and memory
            self.ctx
                .device
                .destroy_buffer(self.cube_vertex_buffer, None);
            self.ctx
                .device
                .free_memory(self.cube_vertex_buffer_memory, None);

            self.ctx.device.destroy_buffer(self.cube_index_buffer, None);
            self.ctx
                .device
                .free_memory(self.cube_index_buffer_memory, None);

            self.ctx.device.destroy_buffer(self.voxel_buffer, None);
            self.ctx.device.free_memory(self.voxel_buffer_memory, None);
            self.ctx.device.destroy_sampler(self.voxel_sampler, None);
            self.ctx
                .device
                .destroy_image_view(self.voxel_image_view, None);
            self.ctx.device.destroy_image(self.voxel_image, None);
            self.ctx.device.free_memory(self.voxel_image_memory, None);

            self.ctx.device.destroy_buffer(self.octree_buffer, None);
            self.ctx.device.free_memory(self.octree_buffer_memory, None);

            // Cleaning up descriptor pool automatically frees descriptor sets
            self.ctx
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.ctx
                    .device
                    .destroy_buffer(self.uniform_buffers[i], None);
                self.ctx
                    .device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            // Destroy graphics pipeline and layout
            self.ctx
                .device
                .destroy_pipeline(self.graphics_pipeline, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            // Destroy framebuffers (one per swapchain image)
            for &framebuffer in &self.swapchain_framebuffers {
                self.ctx.device.destroy_framebuffer(framebuffer, None);
            }

            // Destroy image views (one per swapchain image)
            for &image_view in &self.swapchain_image_views {
                self.ctx.device.destroy_image_view(image_view, None);
            }

            // Destroy render pass
            self.ctx.device.destroy_render_pass(self.render_pass, None);

            // Destroy shader modules
            self.ctx
                .device
                .destroy_shader_module(self.vert_shader_module, None);
            self.ctx
                .device
                .destroy_shader_module(self.frag_shader_module, None);

            // Destroy swapchain (note: images are destroyed automatically)
            self.ctx
                .swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            // Destroy surface
            self.ctx.surface_loader.destroy_surface(self.surface, None);

            // // Destroy egui renderer and state
            // drop(self.egui_renderer.lock().take());
        }
    }

    unsafe fn create_voxel_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: &ash::vk::PhysicalDevice,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer = unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(CHUNK_BYTES as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            device.create_buffer(&info, None)?
        };

        let memory = unsafe {
            let mem_requirements = device.get_buffer_memory_requirements(buffer);
            let memory_type_index = Self::find_memory_type(
                instance,
                physical_device.clone(),
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(CHUNK_BYTES as u64)
                .memory_type_index(memory_type_index);

            device.allocate_memory(&info, None)?
        };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)?;
        }

        let chunk = Chunk::new_sphere(ChunkCoords::new(0, 0, 0), CHUNK_SIZE as f32 / 2.0);

        let data_ptr = unsafe {
            device.map_memory(memory, 0, CHUNK_BYTES as u64, vk::MemoryMapFlags::empty())?
                as *mut u8
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.data().as_ptr() as *const u8,
                data_ptr,
                CHUNK_BYTES,
            );

            device.unmap_memory(memory);
        }

        Ok((buffer, memory))
    }

    unsafe fn create_voxel_texture(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: &ash::vk::PhysicalDevice,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)> {
        let image = unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_3D)
                .format(vk::Format::R8_UINT)
                // .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .mip_levels(1)
                .array_layers(1)
                .extent(vk::Extent3D {
                    width: CHUNK_SIZE,
                    height: CHUNK_SIZE,
                    depth: CHUNK_SIZE,
                });
            device.create_image(&info, None)?
        };

        let memory = unsafe {
            let mem_requirements = device.get_image_memory_requirements(image);
            let memory_type_index = Self::find_memory_type(
                instance,
                physical_device.clone(),
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .context("Failed to find suitable memory type for voxel texture")?;

            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            device.allocate_memory(&info, None)?
        };

        unsafe {
            device.bind_image_memory(image, memory, 0)?;
        }

        let view = unsafe {
            let info = vk::ImageViewCreateInfo::default()
                .image(image.clone())
                .view_type(vk::ImageViewType::TYPE_3D)
                .format(vk::Format::R8_UINT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            device.create_image_view(&info, None)?
        };

        let sampler = unsafe {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

            device.create_sampler(&info, None)?
        };

        Ok((image, memory, view, sampler))
    }

    unsafe fn draw_frame(
        &self,
        debug: &DebugState,
        camera: &Camera,
        camera_transform: &Transform,
    ) -> anyhow::Result<()> {
        let start_time = std::time::Instant::now();

        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            self.ctx.device.wait_for_fences(
                &[self.in_flight_fences[current_frame]],
                true,
                u64::MAX,
            )?;
        }

        // The current frame's fence is now signaled, so any image that was using this fence is now
        // free.
        for fence in self.images_in_flight.iter() {
            let current = *fence.read();
            if current == self.in_flight_fences[current_frame] {
                *fence.write() = vk::Fence::null();
            }
        }

        // Get next image from swapchain
        let (image_index, _is_suboptimal) = unsafe {
            self.ctx.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                // Signal this when ready
                self.image_available_semaphores[current_frame],
                // Don't use fence here
                vk::Fence::null(),
            )?
        };

        {
            let in_flight = *self.images_in_flight[image_index as usize].read();

            // Wait if this image is already in use by another frame
            if in_flight != vk::Fence::null() {
                unsafe {
                    self.ctx
                        .device
                        .wait_for_fences(&[in_flight], true, u64::MAX)?;
                }
            }
        }

        // Mark as in use by the current frame
        *self.images_in_flight[image_index as usize].write() = self.in_flight_fences[current_frame];

        // Reset fence for next time (only after we know we're using this frame)
        unsafe {
            self.ctx
                .device
                .reset_fences(&[self.in_flight_fences[current_frame]])?
        };

        unsafe { self.update_uniform_buffer(camera, camera_transform) };

        unsafe {
            self.ctx.device.reset_command_buffer(
                self.command_buffers[current_frame],
                vk::CommandBufferResetFlags::empty(),
            )?
        };

        // Step 4: Record commands
        self.record_command_buffer(self.command_buffers[current_frame], image_index)?;

        // Step 5: Submit commands to GPU
        unsafe { self.submit_commands(image_index)? };

        // Step 6: Present the result
        unsafe { self.present_image(image_index)? };

        // self.egui_renderer
        //     .lock()
        //     .as_mut()
        //     .unwrap()
        //     .free_textures(&ui.textures_delta.free)?;

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

    unsafe fn create_octree_buffers(
        ctx: &Arc<context::VkContext>,
        command_pool: &vk::CommandPool,
        octree: &Octree,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size =
            (std::mem::size_of::<OctreeNode>() * octree.node_count().max(1)) as vk::DeviceSize;

        let (staging, staging_mem) = unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = ctx.device.create_buffer(&buffer_info, None)?;

            let mem_requirements = ctx.device.get_buffer_memory_requirements(buffer);
            let memory_type_index = Self::find_memory_type(
                &ctx.instance,
                ctx.physical_device.clone(),
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            let mem = ctx.device.allocate_memory(&alloc_info, None)?;

            ctx.device.bind_buffer_memory(buffer, mem, 0)?;

            let data_ptr =
                ctx.device
                    .map_memory(mem, 0, buffer_size, vk::MemoryMapFlags::empty())?
                    as *mut OctreeNode;

            std::ptr::copy_nonoverlapping(octree.nodes().as_ptr(), data_ptr, octree.node_count());

            ctx.device.unmap_memory(mem);

            (buffer, mem)
        };

        let (buf, mem) = unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = ctx.device.create_buffer(&buffer_info, None)?;

            let mem_requirements = ctx.device.get_buffer_memory_requirements(buffer);
            let memory_type_index = Self::find_memory_type(
                &ctx.instance,
                ctx.physical_device.clone(),
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            let mem = ctx.device.allocate_memory(&alloc_info, None)?;

            ctx.device.bind_buffer_memory(buffer, mem, 0)?;

            (buffer, mem)
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool.clone())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };

        if cmd_buffers.len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected to allocate exactly 1 command buffer, got {}",
                cmd_buffers.len()
            ));
        }

        let command_buffer = cmd_buffers[0];

        let begin = vk::CommandBufferBeginInfo::default();

        unsafe {
            ctx.device.begin_command_buffer(command_buffer, &begin)?;
        }

        let buffer_ready_barrier = vk::BufferMemoryBarrier::default()
            .buffer(buf)
            .size(buffer_size)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        unsafe {
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_ready_barrier],
                &[],
            );
        }

        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(buffer_size);

        unsafe {
            ctx.device
                .cmd_copy_buffer(command_buffer, staging, buf, &[region]);
        }

        let buffer_write_barrier = vk::BufferMemoryBarrier::default()
            .buffer(buf)
            .size(buffer_size)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_write_barrier],
                &[],
            );

            ctx.device.end_command_buffer(command_buffer)?;
        }

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            ctx.device
                .queue_submit(ctx.graphics_queue, &[submit_info], vk::Fence::null())?;

            ctx.device.queue_wait_idle(ctx.graphics_queue)?;

            ctx.device
                .free_command_buffers(*command_pool, &[command_buffer]);

            ctx.device.destroy_buffer(staging, None);
            ctx.device.free_memory(staging_mem, None);
        }

        Ok((buf, mem))
    }

    unsafe fn update_uniform_buffer(&self, camera: &Camera, camera_transform: &Transform) {
        let aspect_ratio = self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32;

        let view = camera_transform.matrix().inverse();
        let projection = Mat4::perspective_rh(camera.fov, aspect_ratio, camera.near, camera.far);

        let ubo = UniformBufferObject {
            view,
            projection,
            camera_position: camera_transform.position.extend(0.),
            resolution: Vec3::new(
                self.swapchain_extent.width as f32,
                self.swapchain_extent.height as f32,
                aspect_ratio,
            )
            .extend(0.),
        };

        let current_image = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        // Copy to GPU memory
        let data_ptr = unsafe {
            self.ctx
                .device
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

            self.ctx
                .device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }

    unsafe fn recreate_swapchain(&self, ctx: &Arc<context::VkContext>) -> anyhow::Result<()> {
        unsafe {
            self.ctx.device.device_wait_idle()?;

            for &fb in &self.swapchain_framebuffers {
                self.ctx.device.destroy_framebuffer(fb, None);
            }

            for &view in &self.swapchain_image_views {
                self.ctx.device.destroy_image_view(view, None);
            }

            ctx.swapchain_loader.destroy_swapchain(self.swapchain, None);

            let (swapchain, _, surface_format, extent) = Self::create_swap_chain(
                &ctx.instance,
                &ctx.physical_device,
                &ctx.device,
                &self.surface,
                &ctx.surface_loader,
                &self.window,
            )?;

            let swapchain_images = ctx.swapchain_loader.get_swapchain_images(swapchain)?;

            let (image_views, framebuffers) = Self::create_framebuffers(
                &ctx.device,
                &swapchain_images,
                self.render_pass,
                extent,
                surface_format.format,
            )?;

            let self_mut = self as *const Self as *mut Self;
            (*self_mut).swapchain = swapchain;
            (*self_mut).swapchain_extent = extent;
            (*self_mut).swapchain_format = surface_format.format;
            (*self_mut).swapchain_framebuffers = framebuffers;
            (*self_mut).swapchain_image_views = image_views;
            (*self_mut).swapchain_images = swapchain_images;

            tracing::debug!("Main window swapchain recreated with extent {:?}", extent);
        }

        Ok(())
    }

    unsafe fn submit_commands(&self, image_index: u32) -> anyhow::Result<()> {
        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);
        let wait_semaphores = [self.image_available_semaphores[current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];
        let command_buffers = [self.command_buffers[current_frame]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores) // Don't start until image is available
            .wait_dst_stage_mask(&wait_stages) // Wait specifically before color output
            .command_buffers(&command_buffers) // The commands to execute
            .signal_semaphores(&signal_semaphores); // Signal when rendering is done

        unsafe {
            self.ctx.device.queue_submit(
                self.ctx.graphics_queue,
                &[submit_info],
                self.in_flight_fences[current_frame], // Signal fence when GPU work is done
            )?
        };

        Ok(())
    }

    unsafe fn present_image(&self, image_index: u32) -> anyhow::Result<()> {
        let wait_semaphores = [self.render_finished_semaphores[image_index as usize]];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores) // Don't present until rendering is done
            .swapchains(&swapchains) // Which swapchain to present to
            .image_indices(&image_indices); // Which image in that swapchain

        unsafe {
            self.ctx
                .swapchain_loader
                .queue_present(self.ctx.graphics_queue, &present_info)?
        };

        Ok(())
    }

    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_idx: u32,
    ) -> anyhow::Result<()> {
        let current_frame = self.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        // Start recording
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.ctx
                .device
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
            self.ctx.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE, // We'll record commands directly (not secondary command buffers)
            )
        };

        // Bind graphics pipeline
        unsafe {
            self.ctx.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        let swapchain_extent = self.swapchain_extent;

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

        unsafe {
            self.ctx
                .device
                .cmd_set_viewport(command_buffer, 0, &viewports);
            self.ctx
                .device
                .cmd_set_scissor(command_buffer, 0, &scissors);
        }

        // Bind vertex buffer
        let vertex_buffers = [self.cube_vertex_buffer];
        let offsets = [0];
        unsafe {
            self.ctx
                .device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

            self.ctx.device.cmd_bind_index_buffer(
                command_buffer,
                self.cube_index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            self.ctx.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[current_frame]],
                &[],
            );

            // Draw fullscreen quad once - no transforms needed for raymarching!
            self.ctx.device.cmd_draw_indexed(
                command_buffer,
                crate::shapes::FULL_SCREEN_QUAD_INDICES.len() as u32,
                1, // instance count
                0, // first index
                0, // vertex offset
                0, // first instance
            );
        }

        // End render pass
        unsafe { self.ctx.device.cmd_end_render_pass(command_buffer) };

        // Finish recording
        unsafe { self.ctx.device.end_command_buffer(command_buffer)? };

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

        let window_size = window.inner_size();
        tracing::debug!(
            "Main window: inner_size={:?}, scale_factor={}",
            window_size,
            window.scale_factor()
        );
        tracing::debug!(
            "Main window: capabilities.current_extent={:?}",
            capabilities.current_extent
        );

        let extent = if capabilities.current_extent.width != u32::MAX {
            // Surface tells us exactly what size to use
            capabilities.current_extent
        } else {
            // We need to figure out the size from the window

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

        tracing::debug!("Main window: final extent={:?}", extent);

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
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let voxel_texture_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1) // binding = 1 in shader
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [ubo_layout_binding, voxel_texture_binding];
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
        voxel_image_view: vk::ImageView,
        voxel_sampler: vk::Sampler,
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

        for descriptor_set in &descriptor_sets {
            unsafe {
                let info = vk::DescriptorImageInfo::default()
                    .image_view(voxel_image_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(voxel_sampler);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set) // All sets use the same texture
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&info));

                device.update_descriptor_sets(&[write], &[]);
            }
        }

        tracing::info!("Descriptor sets created");

        Ok(descriptor_sets)
    }

    unsafe fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let uniform_buffer_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

        let voxel_texture_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

        let pool_sizes = [uniform_buffer_size, voxel_texture_size];

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

        tracing::info!("Uniform buffers created");

        Ok((buffers, buffers_memory))
    }

    unsafe fn create_index_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = (std::mem::size_of::<u16>()
            * crate::shapes::FULL_SCREEN_QUAD_INDICES.len())
            as vk::DeviceSize;

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
                crate::shapes::FULL_SCREEN_QUAD_INDICES.as_ptr(),
                data_ptr,
                crate::shapes::FULL_SCREEN_QUAD_INDICES.len(),
            )
        };

        unsafe { device.unmap_memory(buffer_memory) };

        tracing::info!(
            "Index buffer created with {} indices",
            crate::shapes::FULL_SCREEN_QUAD_INDICES.len()
        );

        Ok((buffer, buffer_memory))
    }

    unsafe fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> anyhow::Result<ash::vk::SurfaceKHR> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.into(),
                window.window_handle()?.into(),
                None,
            )?
        };

        Ok(surface)
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
        swapchain_image_count: usize,
    ) -> anyhow::Result<(
        Vec<vk::Semaphore>,
        Vec<vk::Semaphore>,
        Vec<vk::Fence>,
        Vec<RwLock<vk::Fence>>,
    )> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // Start signaled so first frame doesn't wait
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .context("Failed to create image available semaphore")?,
                );
                in_flight_fences.push(
                    device
                        .create_fence(&fence_info, None)
                        .context("Failed to create in-flight fence")?,
                );
            }
        }

        // let mut render_finished_
        for _ in 0..swapchain_image_count {
            unsafe {
                render_finished_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .context("Failed to create render finished semaphore")?,
                );
            }
        }

        let images_in_flight = Vec::from_iter(
            std::iter::from_fn(|| Some(RwLock::new(vk::Fence::null()))).take(swapchain_image_count),
        );

        Ok((
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
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
        let buffer_size = (std::mem::size_of::<Vertex>()
            * crate::shapes::FULL_SCREEN_QUAD_VERTICES.len())
            as vk::DeviceSize;

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
                crate::shapes::FULL_SCREEN_QUAD_VERTICES.as_ptr(),
                data_ptr,
                crate::shapes::FULL_SCREEN_QUAD_VERTICES.len(),
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
            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(std::mem::size_of::<Mat4>() as u32);

            let set_layouts = [descriptor_set_layout];
            let push_constant_ranges = [push_constant_range];

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges);

            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create pipeline layout")?
        };

        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        // 9. Create the graphics pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .dynamic_state(&dynamic_state_info) // No dynamic state for
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

        tracing::info!("Graphics pipeline created");

        Ok((graphics_pipeline, pipeline_layout))
    }
}
