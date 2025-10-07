use std::{cell::OnceCell, sync::Arc};

use ash::vk;
use egui::ViewportBuilder;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event_loop::ActiveEventLoop;

use crate::renderer::vulkan::context::VkContext;

struct DebugWindowInner {
    window: Arc<winit::window::Window>,
    state: egui_winit::State,

    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
    swapchain_format: vk::Format,

    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    current_frame: std::sync::atomic::AtomicUsize,

    renderer: egui_ash_renderer::Renderer,
}

pub struct DebugWindow {
    vk_ctx: Arc<VkContext>,
    ctx: egui::Context,
    inner: OnceCell<DebugWindowInner>,
}

impl DebugWindow {
    pub fn new(vk_ctx: Arc<VkContext>) -> Self {
        Self {
            vk_ctx,
            ctx: egui::Context::default(),
            inner: OnceCell::new(),
        }
    }

    pub unsafe fn initialize(&mut self, ev: &ActiveEventLoop) -> anyhow::Result<()> {
        let viewport = ViewportBuilder::default()
            .with_visible(true)
            .with_title("RVoxel Debug");

        let window = Arc::new(egui_winit::create_window(&self.ctx, ev, &viewport)?);

        // Create Vulkan surface for this window
        let surface = unsafe {
            ash_window::create_surface(
                &self.vk_ctx.entry,
                &self.vk_ctx.instance,
                window.display_handle()?.into(),
                window.window_handle()?.into(),
                None,
            )?
        };

        // Create swapchain (reuse logic from main renderer)
        let (swapchain, surface_format, extent) = unsafe {
            self.create_swapchain(&surface, &window)?
        };

        let actual_window_size = window.inner_size();
        tracing::debug!("Debug window: after swapchain creation - window inner_size={:?}, swapchain extent={:?}",
            actual_window_size, extent);

        let swapchain_images = unsafe {
            self.vk_ctx
                .swapchain_loader
                .get_swapchain_images(swapchain)?
        };

        // Create render pass
        let render_pass = unsafe { self.create_render_pass(surface_format.format)? };

        // Create image views and framebuffers
        let (image_views, framebuffers) = unsafe {
            self.create_framebuffers(
                &swapchain_images,
                render_pass,
                extent,
                surface_format.format,
            )?
        };

        // Create command pool and buffers
        let command_pool = unsafe { self.create_command_pool()? };

        let command_buffers = unsafe { self.allocate_command_buffers(command_pool)? };

        // Create synchronization objects (one pair per swapchain image)
        let (image_available_semaphores, render_finished_semaphores) = unsafe {
            self.create_semaphores(swapchain_images.len())?
        };

        // Create egui renderer
        let renderer = egui_ash_renderer::Renderer::with_default_allocator(
            &self.vk_ctx.instance,
            self.vk_ctx.physical_device,
            self.vk_ctx.device.clone(),
            render_pass,
            egui_ash_renderer::Options::default(),
        )?;

        let state = egui_winit::State::new(
            self.ctx.clone(),
            self.ctx.viewport_id(),
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let inner = DebugWindowInner {
            window,
            state,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views: image_views,
            swapchain_extent: extent,
            swapchain_format: surface_format.format,
            render_pass,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            current_frame: std::sync::atomic::AtomicUsize::new(0),
            renderer,
        };

        self.inner
            .set(inner)
            .map_err(|_| ())
            .expect("should only be initialized once");

        Ok(())
    }

    unsafe fn create_swapchain(
        &self,
        surface: &vk::SurfaceKHR,
        window: &winit::window::Window,
    ) -> anyhow::Result<(vk::SwapchainKHR, vk::SurfaceFormatKHR, vk::Extent2D)> {
        let capabilities = unsafe {
            self.vk_ctx
                .surface_loader
                .get_physical_device_surface_capabilities(self.vk_ctx.physical_device, *surface)?
        };

        let formats = unsafe {
            self.vk_ctx
                .surface_loader
                .get_physical_device_surface_formats(self.vk_ctx.physical_device, *surface)?
        };

        let surface_format = formats[0];

        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
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

        tracing::debug!("Debug window: final extent={:?}", extent);

        let image_count =
            (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
                capabilities.max_image_count
            } else {
                u32::MAX
            });

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = unsafe {
            self.vk_ctx
                .swapchain_loader
                .create_swapchain(&create_info, None)?
        };

        Ok((swapchain, surface_format, extent))
    }

    unsafe fn create_render_pass(&self, format: vk::Format) -> anyhow::Result<vk::RenderPass> {
        use anyhow::Context;

        let attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass));

        unsafe {
            self.vk_ctx
                .device
                .create_render_pass(&create_info, None)
                .context("Failed to create render pass")
        }
    }

    unsafe fn create_framebuffers(
        &self,
        images: &[vk::Image],
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> anyhow::Result<(Vec<vk::ImageView>, Vec<vk::Framebuffer>)> {
        use anyhow::Context;

        let image_views: Vec<_> = images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe {
                    self.vk_ctx
                        .device
                        .create_image_view(&create_info, None)
                        .context("Failed to create image view")
                }
            })
            .collect::<Result<_, _>>()?;

        let framebuffers: Vec<_> = image_views
            .iter()
            .map(|&view| {
                let attachments = [view];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe {
                    self.vk_ctx
                        .device
                        .create_framebuffer(&create_info, None)
                        .context("Failed to create framebuffer")
                }
            })
            .collect::<Result<_, _>>()?;

        Ok((image_views, framebuffers))
    }

    unsafe fn create_command_pool(&self) -> anyhow::Result<vk::CommandPool> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.vk_ctx.graphics_queue_family_index);

        Ok(unsafe { self.vk_ctx.device.create_command_pool(&create_info, None)? })
    }

    unsafe fn allocate_command_buffers(
        &self,
        pool: vk::CommandPool,
    ) -> anyhow::Result<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        Ok(unsafe { self.vk_ctx.device.allocate_command_buffers(&alloc_info)? })
    }

    unsafe fn create_semaphores(&self, count: usize) -> anyhow::Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>)> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        let mut image_available = Vec::with_capacity(count);
        let mut render_finished = Vec::with_capacity(count);

        for _ in 0..count {
            image_available.push(unsafe {
                self.vk_ctx.device.create_semaphore(&semaphore_info, None)?
            });

            render_finished.push(unsafe {
                self.vk_ctx.device.create_semaphore(&semaphore_info, None)?
            });
        }

        Ok((image_available, render_finished))
    }

    pub fn window(&self) -> &Arc<winit::window::Window> {
        &self.inner.get().expect("DebugWindow not initialized").window
    }

    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    pub fn draw(&mut self, draw_fn: impl FnMut(&egui::Context)) -> egui::FullOutput {
        let inner = self.inner.get_mut().expect("DebugWindow not initialized");
        let input = inner.state.take_egui_input(inner.window.as_ref());

        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::debug!("Debug window egui: screen_rect={:?}, viewport={:?}",
                input.screen_rect, input.viewports.get(&input.viewport_id));
        });

        let output = self.ctx.run(input, draw_fn);
        inner.state.handle_platform_output(inner.window.as_ref(), output.platform_output.clone());
        output
    }

    pub unsafe fn render(&mut self, draw_fn: impl FnMut(&egui::Context)) -> anyhow::Result<()> {
        let output = self.draw(draw_fn);
        let inner = self.inner.get_mut().expect("DebugWindow not initialized");

        let textures_to_set: Vec<_> = output
            .textures_delta
            .set
            .iter()
            .map(|(id, delta)| (*id, delta.clone()))
            .collect();

        if !textures_to_set.is_empty() {
            inner.renderer.set_textures(self.vk_ctx.graphics_queue, inner.command_pool, &textures_to_set)?;
        }

        let current_frame = inner.current_frame.load(std::sync::atomic::Ordering::SeqCst);

        let (image_index, _) = unsafe {
            self.vk_ctx.swapchain_loader.acquire_next_image(
                inner.swapchain,
                u64::MAX,
                inner.image_available_semaphores[current_frame],
                vk::Fence::null(),
            )?
        };

        let image_available_semaphore = inner.image_available_semaphores[current_frame];
        let render_finished_semaphore = inner.render_finished_semaphores[current_frame];

        unsafe {
            let command_buffer = inner.command_buffers[0];

            self.vk_ctx
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.vk_ctx
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            };

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(inner.render_pass)
                .framebuffer(inner.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: inner.swapchain_extent,
                })
                .clear_values(std::slice::from_ref(&clear_value));

            self.vk_ctx.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );

            inner.renderer.cmd_draw(
                command_buffer,
                inner.swapchain_extent,
                output.pixels_per_point,
                &self.ctx.tessellate(output.shapes, output.pixels_per_point),
            )?;

            self.vk_ctx.device.cmd_end_render_pass(command_buffer);
            self.vk_ctx.device.end_command_buffer(command_buffer)?;

            // Submit
            let wait_semaphores = [image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [render_finished_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(&command_buffer))
                .signal_semaphores(&signal_semaphores);

            self.vk_ctx.device.queue_submit(
                self.vk_ctx.graphics_queue,
                &[submit_info],
                vk::Fence::null(),
            )?;

            self.vk_ctx
                .device
                .queue_wait_idle(self.vk_ctx.graphics_queue)?;

            let wait_semaphores = [render_finished_semaphore];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(std::slice::from_ref(&inner.swapchain))
                .image_indices(std::slice::from_ref(&image_index));

            self.vk_ctx
                .swapchain_loader
                .queue_present(self.vk_ctx.graphics_queue, &present_info)?;
        }

        if !output.textures_delta.free.is_empty() {
            inner.renderer.free_textures(&output.textures_delta.free)?;
        }

        inner.current_frame.store(
            (current_frame + 1) % inner.image_available_semaphores.len(),
            std::sync::atomic::Ordering::SeqCst,
        );

        Ok(())
    }

    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) -> egui_winit::EventResponse {
        if let winit::event::WindowEvent::Resized(new_size) = event {
            if new_size.width > 0 && new_size.height > 0 {
                tracing::debug!("Debug window resized to {:?}", new_size);
                unsafe {
                    if let Err(e) = self.recreate_swapchain() {
                        tracing::error!("Failed to recreate swapchain: {}", e);
                    }
                }
            }
        }

        let inner = self.inner.get_mut().expect("DebugWindow not initialized");
        inner.state.on_window_event(inner.window.as_ref(), event)
    }

    unsafe fn recreate_swapchain(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.vk_ctx.device.device_wait_idle()?;

            let inner = self.inner.get_mut().expect("DebugWindow not initialized");

            for &fb in &inner.framebuffers {
                self.vk_ctx.device.destroy_framebuffer(fb, None);
            }

            for &view in &inner.swapchain_image_views {
                self.vk_ctx.device.destroy_image_view(view, None);
            }

            self.vk_ctx.swapchain_loader.destroy_swapchain(inner.swapchain, None);

            let surface = inner.surface;
            let window = Arc::clone(&inner.window);
            let render_pass = inner.render_pass;

            drop(inner);

            let (swapchain, surface_format, extent) = self.create_swapchain(&surface, &window)?;

            let swapchain_images = self.vk_ctx.swapchain_loader.get_swapchain_images(swapchain)?;

            let (image_views, framebuffers) = self.create_framebuffers(
                &swapchain_images,
                render_pass,
                extent,
                surface_format.format,
            )?;

            let inner = self.inner.get_mut().expect("DebugWindow not initialized");
            inner.swapchain = swapchain;
            inner.swapchain_images = swapchain_images;
            inner.swapchain_image_views = image_views;
            inner.swapchain_extent = extent;
            inner.swapchain_format = surface_format.format;
            inner.framebuffers = framebuffers;

            tracing::debug!("Debug window swapchain recreated with extent {:?}", extent);
        }

        Ok(())
    }

    pub unsafe fn cleanup(&mut self) {
        if let Some(inner) = self.inner.take() {
            unsafe {
                self.vk_ctx.device.device_wait_idle().ok();

                // Destroy renderer first (owns Vulkan resources)
                drop(inner.renderer);

                for &semaphore in &inner.image_available_semaphores {
                    self.vk_ctx.device.destroy_semaphore(semaphore, None);
                }

                for &semaphore in &inner.render_finished_semaphores {
                    self.vk_ctx.device.destroy_semaphore(semaphore, None);
                }

                self.vk_ctx.device.destroy_command_pool(inner.command_pool, None);

                for &fb in &inner.framebuffers {
                    self.vk_ctx.device.destroy_framebuffer(fb, None);
                }

                for &view in &inner.swapchain_image_views {
                    self.vk_ctx.device.destroy_image_view(view, None);
                }

                self.vk_ctx.device.destroy_render_pass(inner.render_pass, None);

                self.vk_ctx
                    .swapchain_loader
                    .destroy_swapchain(inner.swapchain, None);

                self.vk_ctx.surface_loader.destroy_surface(inner.surface, None);
            }
        }
    }
}
