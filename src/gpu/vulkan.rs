//! Minimal ash device + swapchain + compute-fill path for Week 1.

use anyhow::{Context, Result, anyhow, bail};
use ash::khr::{portability_enumeration, surface, swapchain};
use ash::{Device, Entry, Instance, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::{CStr, CString};
use std::mem::size_of;
use std::path::Path;
use std::time::Instant;
use winit::window::Window;

const QUEUE_PRIORITY: f32 = 1.0;

fn load_vulkan_entry() -> Result<Entry> {
    if let Ok(entry) = unsafe { Entry::load() } {
        return Ok(entry);
    }
    // Homebrew / LunarG locations (macOS does not put libvulkan on the default dyld path).
    let candidates = [
        std::env::var_os("VULKAN_LIB").map(std::path::PathBuf::from),
        Some("/opt/homebrew/lib/libvulkan.dylib".into()),
        Some("/usr/local/lib/libvulkan.dylib".into()),
        std::env::var_os("VULKAN_SDK").map(|sdk| Path::new(&sdk).join("lib/libvulkan.dylib")),
    ];
    let mut last = None;
    for path in candidates.into_iter().flatten() {
        match unsafe { Entry::load_from(&path) } {
            Ok(entry) => return Ok(entry),
            Err(err) => last = Some((path, err)),
        }
    }
    match last {
        Some((path, err)) => Err(err).with_context(|| format!("also tried {}", path.display())),
        None => bail!("no Vulkan loader candidates"),
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FillPush {
    resolution: [f32; 2],
    time: f32,
    _pad: f32,
}

pub struct VulkanContext {
    _entry: Entry,
    pub instance: Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    pub device: Device,
    queue: vk::Queue,
    queue_family: u32,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame_index: usize,
    storage_image: vk::Image,
    storage_memory: vk::DeviceMemory,
    storage_view: vk::ImageView,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,
    start: Instant,
}

impl VulkanContext {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = load_vulkan_entry().context("load Vulkan loader (MoltenVK / libvulkan)")?;

        let display = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        // Keep CStrings alive for the create_* calls below.
        let get_phys_props2 = CString::new("VK_KHR_get_physical_device_properties2")?;
        let portability_subset = CString::new("VK_KHR_portability_subset")?;

        let mut extension_names =
            ash_window::enumerate_required_extensions(display)?.to_vec();
        extension_names.push(surface::NAME.as_ptr());
        // MoltenVK is a portability driver — required to see any physical device.
        extension_names.push(portability_enumeration::NAME.as_ptr());
        extension_names.push(get_phys_props2.as_ptr());

        let app_name = c"helmholtz-wods";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("create Vulkan instance")?;

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display, window_handle, None)
        }
        .context("create surface")?;
        let surface_loader = surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family) =
            pick_device(&instance, &surface_loader, surface)?;

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(std::slice::from_ref(&QUEUE_PRIORITY));

        // Keep device creation conservative for MoltenVK; enable sync2 later (W2 timestamps).
        let device_extensions = [swapchain::NAME.as_ptr(), portability_subset.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("create logical device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        // Extension name CStrings must remain alive through create_instance / create_device.
        let _keep_ext_names = (&get_phys_props2, &portability_subset);
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let size = window.inner_size();
        let (swapchain, swapchain_format, swapchain_extent, swapchain_images) = create_swapchain(
            &instance,
            &device,
            &surface_loader,
            &swapchain_loader,
            surface,
            physical_device,
            queue_family,
            vk::Extent2D {
                width: size.width.max(1),
                height: size.height.max(1),
            },
            None,
        )?;

        let swapchain_views = create_image_views(&device, &swapchain_images, swapchain_format)?;

        let (storage_image, storage_memory, storage_view) =
            create_storage_image(&instance, &device, physical_device, swapchain_extent)?;

        let descriptor_set_layout = create_descriptor_set_layout(&device)?;
        let pipeline_layout = create_pipeline_layout(&device, descriptor_set_layout)?;
        let compute_pipeline = create_compute_pipeline(&device, pipeline_layout)?;
        let (descriptor_pool, descriptor_set) =
            create_descriptors(&device, descriptor_set_layout, storage_view)?;

        let command_pool = {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { device.create_command_pool(&info, None) }?
        };

        let frames = swapchain_images.len();
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc) }?;

        let mut image_available = Vec::with_capacity(frames);
        let mut render_finished = Vec::with_capacity(frames);
        let mut in_flight = Vec::with_capacity(frames);
        for _ in 0..frames {
            image_available.push(unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            });
            render_finished.push(unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            });
            in_flight.push(unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?
            });
        }

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            queue_family,
            swapchain_loader,
            swapchain,
            swapchain_format,
            swapchain_extent,
            swapchain_images,
            swapchain_views,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_index: 0,
            storage_image,
            storage_memory,
            storage_view,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            compute_pipeline,
            start: Instant::now(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        unsafe { self.device.device_wait_idle()? };
        self.destroy_swapchain_resources();

        let (swapchain, format, extent, images) = create_swapchain(
            &self.instance,
            &self.device,
            &self.surface_loader,
            &self.swapchain_loader,
            self.surface,
            self.physical_device,
            self.queue_family,
            vk::Extent2D { width, height },
            None,
        )?;
        self.swapchain = swapchain;
        self.swapchain_format = format;
        self.swapchain_extent = extent;
        self.swapchain_images = images;
        self.swapchain_views =
            create_image_views(&self.device, &self.swapchain_images, self.swapchain_format)?;

        let (img, mem, view) =
            create_storage_image(&self.instance, &self.device, self.physical_device, extent)?;
        self.storage_image = img;
        self.storage_memory = mem;
        self.storage_view = view;

        unsafe {
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
        let (pool, set) =
            create_descriptors(&self.device, self.descriptor_set_layout, self.storage_view)?;
        self.descriptor_pool = pool;
        self.descriptor_set = set;

        let frames = self.swapchain_images.len();
        if self.command_buffers.len() != frames {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
            }
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(frames as u32);
            self.command_buffers = unsafe { self.device.allocate_command_buffers(&alloc)? };

            for sem in self.image_available.drain(..) {
                unsafe { self.device.destroy_semaphore(sem, None) };
            }
            for sem in self.render_finished.drain(..) {
                unsafe { self.device.destroy_semaphore(sem, None) };
            }
            for fence in self.in_flight.drain(..) {
                unsafe { self.device.destroy_fence(fence, None) };
            }
            for _ in 0..frames {
                self.image_available.push(unsafe {
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
                });
                self.render_finished.push(unsafe {
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
                });
                self.in_flight.push(unsafe {
                    self.device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )?
                });
            }
        }
        Ok(())
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        let fi = self.frame_index % self.in_flight.len();
        let fence = self.in_flight[fi];
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
            self.device.reset_fences(std::slice::from_ref(&fence))?;
        }

        let (image_index, _suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[fi],
                vk::Fence::null(),
            )?
        };
        let image_index = image_index as usize;

        let cmd = self.command_buffers[fi];
        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        }

        // Storage image: UNDEFINED → GENERAL for compute write.
        cmd_image_barrier(
            &self.device,
            cmd,
            self.storage_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );

        let push = FillPush {
            resolution: [
                self.swapchain_extent.width as f32,
                self.swapchain_extent.height as f32,
            ],
            time: self.start.elapsed().as_secs_f32(),
            _pad: 0.0,
        };

        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            let gx = (self.swapchain_extent.width + 7) / 8;
            let gy = (self.swapchain_extent.height + 7) / 8;
            self.device.cmd_dispatch(cmd, gx, gy, 1);
        }

        // Storage GENERAL → TRANSFER_SRC; swapchain → TRANSFER_DST; blit; present.
        cmd_image_barrier(
            &self.device,
            cmd,
            self.storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        );
        cmd_image_barrier(
            &self.device,
            cmd,
            self.swapchain_images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::ImageBlit {
            src_subresource: subresource,
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.swapchain_extent.width as i32,
                    y: self.swapchain_extent.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: subresource,
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.swapchain_extent.width as i32,
                    y: self.swapchain_extent.height as i32,
                    z: 1,
                },
            ],
        };
        unsafe {
            self.device.cmd_blit_image(
                cmd,
                self.storage_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.swapchain_images[image_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
                vk::Filter::NEAREST,
            );
        }

        cmd_image_barrier(
            &self.device,
            cmd,
            self.swapchain_images[image_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );

        unsafe { self.device.end_command_buffer(cmd)? };

        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&self.image_available[fi]))
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(std::slice::from_ref(&self.render_finished[fi]));

        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), fence)?;
        }

        let image_index_u32 = image_index as u32;
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&self.render_finished[fi]))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(std::slice::from_ref(&image_index_u32));

        unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present)
                .context("queue_present")?;
        }

        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(())
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.storage_view, None);
            self.device.destroy_image(self.storage_image, None);
            self.device.free_memory(self.storage_memory, None);
            for view in self.swapchain_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for fence in self.in_flight.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            for sem in self.image_available.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            for sem in self.render_finished.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.destroy_swapchain_resources();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    for physical in devices {
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        for (index, props) in families.iter().enumerate() {
            let index = index as u32;
            let graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = props.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let present = unsafe {
                surface_loader.get_physical_device_surface_support(physical, index, surface)?
            };
            if graphics && compute && present {
                return Ok((physical, index));
            }
        }
    }
    bail!("no suitable Vulkan queue family (graphics+compute+present)")
}

fn create_swapchain(
    instance: &Instance,
    _device: &Device,
    surface_loader: &surface::Instance,
    swapchain_loader: &swapchain::Device,
    surface: vk::SurfaceKHR,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    extent_hint: vk::Extent2D,
    old: Option<vk::SwapchainKHR>,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
    let caps = unsafe { surface_loader.get_physical_device_surface_capabilities(physical, surface) }?;
    let formats = unsafe { surface_loader.get_physical_device_surface_formats(physical, surface) }?;
    let present_modes =
        unsafe { surface_loader.get_physical_device_surface_present_modes(physical, surface) }?;

    let surface_format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_UNORM
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .ok_or_else(|| anyhow!("no surface formats"))?;

    // Proposal: present with vsync off when available.
    let present_mode = if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE
    } else if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: extent_hint
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: extent_hint
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 {
        image_count = image_count.min(caps.max_image_count);
    }

    let mut info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);
    let qf = [queue_family];
    info = info.queue_family_indices(&qf);
    if let Some(old_sc) = old {
        info = info.old_swapchain(old_sc);
    }

    let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None) }?;
    let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;
    let _ = instance; // silence if unused on some paths
    Ok((swapchain, surface_format.format, extent, images))
}

fn create_image_views(
    device: &Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|&image| {
            let info = vk::ImageViewCreateInfo::default()
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
            Ok(unsafe { device.create_image_view(&info, None)? })
        })
        .collect()
}

fn create_storage_image(
    instance: &Instance,
    device: &Device,
    physical: vk::PhysicalDevice,
    extent: vk::Extent2D,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&info, None) }?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        instance,
        physical,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }?;
    unsafe { device.bind_image_memory(image, memory, 0)? };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_info, None)? };
    Ok((image, memory, view))
}

fn find_memory_type(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let mem = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..mem.memory_type_count {
        if type_bits & (1 << i) != 0
            && mem.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }
    bail!("no matching memory type")
}

fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);
    let info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
    Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
}

fn create_pipeline_layout(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: size_of::<FillPush>() as u32,
    };
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(std::slice::from_ref(&push));
    Ok(unsafe { device.create_pipeline_layout(&info, None)? })
}

fn load_shader_module(device: &Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    Ok(unsafe { device.create_shader_module(&info, None)? })
}

fn create_compute_pipeline(device: &Device, layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
    let spv_path = env!("SHADER_FILL_COMP");
    let bytes = std::fs::read(spv_path)
        .with_context(|| format!("read compiled SPIR-V at {spv_path}"))?;
    let module = load_shader_module(device, &bytes)?;
    let main_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(main_name);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipelines = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None)
    }
    .map_err(|(_, e)| e)?;
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipelines[0])
}

fn create_descriptors(
    device: &Device,
    layout: vk::DescriptorSetLayout,
    storage_view: vk::ImageView,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_IMAGE,
        descriptor_count: 1,
    };
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(std::slice::from_ref(&pool_size))
        .max_sets(1);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(std::slice::from_ref(&layout));
    let set = unsafe { device.allocate_descriptor_sets(&alloc)? }[0];

    let image_info = vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: storage_view,
        image_layout: vk::ImageLayout::GENERAL,
    };
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(std::slice::from_ref(&image_info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    Ok((pool, set))
}

fn cmd_image_barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}
