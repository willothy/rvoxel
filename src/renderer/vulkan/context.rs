use anyhow::Context;
use ash::khr;
use ash::vk;

pub struct VkContext {
    pub(crate) entry: ash::Entry,

    /// The Vulkan instance.
    ///
    /// TODO: figure out and explain what this really is
    pub(crate) instance: ash::Instance,

    /// Physical device represents a GPU in the system that we have
    /// selected.
    #[allow(unused)]
    pub(crate) physical_device: ash::vk::PhysicalDevice,

    /// The logical device is our connection to the physical device.
    ///
    /// You can think of it like a session between the application and the GPU driver.
    pub(crate) device: ash::Device,

    /// Queue for submitting graphics commands.
    pub(crate) graphics_queue: ash::vk::Queue,

    pub(crate) graphics_queue_family_index: u32,

    pub(crate) surface_loader: khr::surface::Instance,

    pub(crate) swapchain_loader: khr::swapchain::Device,

    #[cfg(all(debug_assertions, feature = "debug"))]
    pub(crate) debug_utils_loader: ash::ext::debug_utils::Instance,
    #[cfg(all(debug_assertions, feature = "debug"))]
    pub(crate) debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VkContext {
    pub fn new() -> anyhow::Result<Self> {
        unsafe {
            let entry = ash::Entry::load().context("failed to load vulkan entry")?;

            let instance =
                Self::create_instance(&entry).context("failed to create vulkan instance")?;

            #[cfg(all(debug_assertions, feature = "debug"))]
            let (debug_utils_loader, debug_messenger) =
                Self::setup_debug_messenger(&entry, &instance)?;

            let physical_device = Self::pick_physical_device(&instance)?;

            let (device, graphics_queue, graphics_queue_family_index) =
                Self::create_logical_device(&instance, &physical_device)?;

            let surface_loader = khr::surface::Instance::new(&entry, &instance);

            let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

            Ok(Self {
                entry,
                instance,
                device,
                physical_device,
                surface_loader,
                swapchain_loader,
                graphics_queue,
                graphics_queue_family_index,

                #[cfg(all(debug_assertions, feature = "debug"))]
                debug_utils_loader,
                #[cfg(all(debug_assertions, feature = "debug"))]
                debug_messenger,
            })
        }
    }

    unsafe fn create_instance(entry: &ash::Entry) -> anyhow::Result<ash::Instance> {
        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(std::ffi::CStr::from_bytes_with_nul(b"rvoxel\0")?)
            .application_version(ash::vk::make_api_version(0, 1, 0, 0))
            .engine_name(std::ffi::CStr::from_bytes_with_nul(b"No Engine\0")?)
            .engine_version(ash::vk::make_api_version(0, 1, 0, 0))
            .api_version(ash::vk::API_VERSION_1_0);

        let extension_names = vec![
            vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
            #[cfg(all(debug_assertions, feature = "debug"))]
            vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        let mut flags = ash::vk::InstanceCreateFlags::default();

        flags |= ash::vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;

        let layer_names = [
            #[cfg(all(debug_assertions, feature = "debug"))]
            ("VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8),
        ];

        let create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names)
            .flags(flags);

        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    #[cfg(all(debug_assertions, feature = "debug"))]
    unsafe fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> anyhow::Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        let debug_utils_loader = ash::ext::debug_utils::Instance::new(entry, instance);

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_messenger = unsafe {
            use anyhow::Context;

            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .context("Failed to create debug messenger")?
        };

        Ok((debug_utils_loader, debug_messenger))
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
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe {
        let callback_data = *p_callback_data;
        if callback_data.p_message.is_null() {
            std::borrow::Cow::from("")
        } else {
            std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        }
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            eprintln!("🔴 VULKAN ERROR [{:?}]: {}", message_type, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            eprintln!("🟡 VULKAN WARNING [{:?}]: {}", message_type, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            println!("🔵 VULKAN INFO [{:?}]: {}", message_type, message);
        }
        _ => {
            println!("⚪ VULKAN [{:?}]: {}", message_type, message);
        }
    }

    vk::FALSE
}
