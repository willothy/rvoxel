use std::{mem::MaybeUninit, sync::atomic::AtomicBool};

use ash::{khr::surface, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{application::ApplicationHandler, event_loop::EventLoop, window::Window};

pub struct VulkanAppInner {
    /// The window that we are rendering to, from [`winit`].
    window: winit::window::Window,

    /// The Vulkan instance.
    ///
    /// TODO: figure out and explain what this really is
    instance: ash::Instance,

    /// Physical device represents a GPU in the system that we have
    /// selected.
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
    swapchain_images: Vec<vk::Image>,
    /// The format of the images in the swapchain.
    swapchain_format: vk::Format,
    /// The extent (width and height) of the swapchain images.
    swapchain_extent: vk::Extent2D,

    command_pool: vk::CommandPool,
    /// Where draw commands are recorded before being submitted to the GPU.
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    current_frame: usize,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanApp {
    app: MaybeUninit<VulkanAppInner>,
    initialized: AtomicBool,
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        if self.is_initialized() {
            unsafe {
                std::ptr::drop_in_place(self.app.as_mut_ptr());
            }
        }
    }
}

impl Drop for VulkanAppInner {
    fn drop(&mut self) {
        // TODO: cleanup more Vulkan resources
        unsafe {
            self.device.device_wait_idle().unwrap();

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.device
                .destroy_command_pool(self.command_pool.clone(), None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain.clone(), None);
        }
    }
}

impl VulkanApp {
    pub fn new_uninit() -> Self {
        Self {
            app: MaybeUninit::uninit(),
            initialized: AtomicBool::new(false),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn initialize(
        &mut self,
        entry: &ash::Entry,
        ev: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        if self.is_initialized() {
            return Ok(());
        }

        let attrs = winit::window::Window::default_attributes()
            .with_content_protected(false)
            .with_title("rvoxel")
            .with_visible(true);

        let window = ev.create_window(attrs)?;

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

        let command_pool = unsafe { Self::create_command_pool(&device, queue_family)? };

        let command_buffers = unsafe { Self::create_command_buffers(&device, &command_pool)? };

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            unsafe { Self::create_sync_objects(&device)? };

        let app_inner = VulkanAppInner {
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
            swapchain_images: unsafe { swapchain_loader.get_swapchain_images(swapchain)? },
            swapchain_loader,

            command_pool,
            command_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,

            current_frame: 0,
        };

        self.app = MaybeUninit::new(app_inner);
        self.initialized
            .store(true, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    pub fn new(
        entry: &ash::Entry,
        ev: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<Self> {
        let mut uninit = Self::new_uninit();

        uninit.initialize(entry, ev)?;

        Ok(uninit)
    }

    pub unsafe fn draw_frame(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            )?;
        }

        // Get next image from swapchain
        let (image_index, _is_suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                // Signal this when ready
                self.image_available_semaphores[self.current_frame],
                // Don't use fence here
                vk::Fence::null(),
            )?
        };

        // Reset fence for next time (only after we know we're using this frame)
        unsafe {
            self.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])?
        };

        // Step 4: Record commands
        self.record_command_buffer(self.command_buffers[self.current_frame], image_index)?;

        // Step 5: Submit commands to GPU
        unsafe { self.submit_commands()? };

        // Step 6: Present the result
        unsafe { self.present_image(image_index)? };

        // Step 7: Move to next frame
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn submit_commands(&self) -> anyhow::Result<()> {
        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let command_buffers = [self.command_buffers[self.current_frame]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores) // Don't start until image is available
            .wait_dst_stage_mask(&wait_stages) // Wait specifically before color output
            .command_buffers(&command_buffers) // The commands to execute
            .signal_semaphores(&signal_semaphores); // Signal when rendering is done

        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame], // Signal fence when GPU work is done
            )?
        };

        Ok(())
    }

    unsafe fn present_image(&self, image_index: u32) -> anyhow::Result<()> {
        let wait_semaphores = [self.render_finished_semaphores[self.current_frame]];
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

    unsafe fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        // Determine pipeline stages and access masks based on layouts
        let (src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
                    // Transitioning from "don't care" to "ready for writing"
                    (
                        vk::PipelineStageFlags::TOP_OF_PIPE, // No previous work to wait for
                        vk::PipelineStageFlags::TRANSFER,    // Transfer operations can start
                        vk::AccessFlags::empty(),            // No previous access
                        vk::AccessFlags::TRANSFER_WRITE,     // Will be writing via transfer
                    )
                }
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => {
                    // Transitioning from "ready for writing" to "ready for display"
                    (
                        vk::PipelineStageFlags::TRANSFER, // Transfer operations must finish
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE, // Before any later operations
                        vk::AccessFlags::TRANSFER_WRITE,  // Was being written to
                        vk::AccessFlags::empty(),         // No specific access needed for present
                    )
                }
                _ => panic!(
                    "Unsupported layout transition: {:?} -> {:?}",
                    old_layout, new_layout
                ),
            };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED) // Not transferring between queues
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR, // Color data (not depth/stencil)
                base_mip_level: 0,                        // Mipmap level 0 (full resolution)
                level_count: 1,                           // Just one mip level
                base_array_layer: 0,                      // Array layer 0 (not array texture)
                layer_count: 1,                           // Just one layer
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,               // Wait for these stages to complete
                dst_stage_mask,               // Before these stages can start
                vk::DependencyFlags::empty(), // No special flags
                &[],                          // No memory barriers
                &[],                          // No buffer barriers
                &[barrier],                   // Our image barrier
            )
        };
    }

    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_idx: u32,
    ) -> anyhow::Result<()> {
        // Start recording
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };

        // For now, just transition image and clear it to blue
        // (We'll add proper rendering here later)
        let image = self.swapchain_images[image_idx as usize];

        // Transition image for clearing
        unsafe {
            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )
        };

        // Clear to blue
        let clear_color = vk::ClearColorValue {
            float32: [0.2, 0.8, 0.8, 1.0], // Nice blue
        };

        unsafe {
            self.device.cmd_clear_color_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            )
        };

        // Transition for presentation
        unsafe {
            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            )
        };

        // Finish recording
        unsafe { self.device.end_command_buffer(command_buffer)? };

        Ok(())
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
}

impl std::ops::Deref for VulkanApp {
    type Target = VulkanAppInner;

    fn deref(&self) -> &Self::Target {
        if !self.initialized.load(std::sync::atomic::Ordering::SeqCst) {
            panic!("Attempted to deref uninitialized VulkanApp");
        }
        unsafe { &*self.app.as_ptr() }
    }
}

impl std::ops::DerefMut for VulkanApp {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if !self.initialized.load(std::sync::atomic::Ordering::SeqCst) {
            panic!("Attempted to deref_mut uninitialized VulkanApp");
        }
        unsafe { &mut *self.app.as_mut_ptr() }
    }
}

pub struct App {
    entry: ash::Entry,

    vk: VulkanApp,
}

impl App {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            entry: unsafe { ash::Entry::load()? },
            vk: VulkanApp::new_uninit(),
        })
    }

    pub fn intialize(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        if self.vk.is_initialized() {
            return Ok(());
        }

        self.vk.initialize(&self.entry, event_loop)?;

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Err(e) = self.intialize(event_loop) {
            eprintln!("Failed to initialize application: {}", e);
        }

        if let Err(e) = unsafe { self.vk.draw_frame() } {
            eprintln!("Error: {e}")
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if !self.vk.is_initialized() {
            return;
        }

        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.vk.window.request_redraw();
                if let Err(e) = unsafe { self.vk.draw_frame() } {
                    eprintln!("Error: {e}")
                }
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new()?;

    event_loop.run_app(&mut app)?;

    Ok(())
}
