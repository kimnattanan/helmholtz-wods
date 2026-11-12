#include "hwods/gpu/vulkan_fill.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hwods {
namespace {

struct FillPush {
  float resolution[2];
  float time;
  float pad;
};

VKAPI_ATTR VkBool32 VKAPI_CALL ignore_messenger(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                VkDebugUtilsMessageTypeFlagsEXT,
                                                const VkDebugUtilsMessengerCallbackDataEXT*,
                                                void*) {
  return VK_FALSE;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f) throw std::runtime_error("failed to read " + path);
  const auto size = static_cast<std::size_t>(f.tellg());
  std::vector<char> buf(size);
  f.seekg(0);
  f.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                          VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mem);
  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) &&
        (mem.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  throw std::runtime_error("no matching memory type");
}

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
                   VkImageLayout new_layout, VkAccessFlags src_access,
                   VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                   VkPipelineStageFlags dst_stage) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = old_layout;
  b.newLayout = new_layout;
  b.srcAccessMask = src_access;
  b.dstAccessMask = dst_access;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace

struct VulkanFillApp::Impl {
  GLFWwindow* window = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D extent{};
  std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_views;

  VkImage storage_image = VK_NULL_HANDLE;
  VkDeviceMemory storage_memory = VK_NULL_HANDLE;
  VkImageView storage_view = VK_NULL_HANDLE;

  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline compute_pipeline = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

  VkCommandPool command_pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkSemaphore> image_available;
  std::vector<VkSemaphore> render_finished;
  std::vector<VkFence> in_flight;
  std::size_t frame_index = 0;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  bool framebuffer_resized = false;

  explicit Impl(int width, int height) {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, "helmholtz-wods — Week 1 compute fill", nullptr,
                              nullptr);
    if (!window) throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int) {
      auto* self = static_cast<Impl*>(glfwGetWindowUserPointer(w));
      self->framebuffer_resized = true;
    });

    create_instance();
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
      throw std::runtime_error("glfwCreateWindowSurface failed");
    }
    pick_device();
    create_device();
    create_swapchain();
    create_storage_and_pipeline();
    create_frames();
  }

  ~Impl() {
    if (device) vkDeviceWaitIdle(device);
    destroy_swapchain_resources();
    if (device) {
      for (auto f : in_flight) vkDestroyFence(device, f, nullptr);
      for (auto s : image_available) vkDestroySemaphore(device, s, nullptr);
      for (auto s : render_finished) vkDestroySemaphore(device, s, nullptr);
      if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
      if (compute_pipeline) vkDestroyPipeline(device, compute_pipeline, nullptr);
      if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
      if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
  }

  void create_instance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "helmholtz-wods";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "helmholtz-wods";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    std::vector<const char*> exts(glfw_exts, glfw_exts + glfw_ext_count);
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateInstance failed — check MoltenVK / libvulkan");
    }
    (void)ignore_messenger;
  }

  void pick_device() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("no Vulkan physical devices (enable portability)");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (auto phys : devices) {
      uint32_t fam_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(phys, &fam_count, nullptr);
      std::vector<VkQueueFamilyProperties> fams(fam_count);
      vkGetPhysicalDeviceQueueFamilyProperties(phys, &fam_count, fams.data());
      for (uint32_t i = 0; i < fam_count; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &present);
        const bool graphics = fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
        const bool compute = fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
        if (graphics && compute && present) {
          physical = phys;
          queue_family = i;
          return;
        }
      }
    }
    throw std::runtime_error("no suitable queue family");
  }

  void create_device() {
    float priority = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* device_exts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset",
    };

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = device_exts;

    if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateDevice failed");
    }
    vkGetDeviceQueue(device, queue_family, 0, &queue);
  }

  void create_swapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);

    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &fmt_count, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats) {
      if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
          f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        chosen = f;
        break;
      }
    }
    swapchain_format = chosen.format;

    uint32_t pm_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &pm_count, nullptr);
    std::vector<VkPresentModeKHR> modes(pm_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &pm_count, modes.data());
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end()) {
      present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    } else if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) !=
               modes.end()) {
      present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
    }

    if (caps.currentExtent.width != UINT32_MAX) {
      extent = caps.currentExtent;
    } else {
      int w = 0, h = 0;
      glfwGetFramebufferSize(window, &w, &h);
      extent.width = std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width,
                                caps.maxImageExtent.width);
      extent.height = std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height,
                                 caps.maxImageExtent.height);
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) image_count = std::min(image_count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = image_count;
    sci.imageFormat = swapchain_format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = present_mode;
    sci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateSwapchainKHR failed");
    }

    uint32_t img_count = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &img_count, nullptr);
    swapchain_images.resize(img_count);
    vkGetSwapchainImagesKHR(device, swapchain, &img_count, swapchain_images.data());

    swapchain_views.resize(img_count);
    for (uint32_t i = 0; i < img_count; ++i) {
      VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vci.image = swapchain_images[i];
      vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vci.format = swapchain_format;
      vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      if (vkCreateImageView(device, &vci, nullptr, &swapchain_views[i]) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateImageView failed");
      }
    }
  }

  void create_storage_image() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {extent.width, extent.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ici, nullptr, &storage_image) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateImage storage failed");
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, storage_image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex =
        find_memory_type(physical, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &mai, nullptr, &storage_memory) != VK_SUCCESS) {
      throw std::runtime_error("vkAllocateMemory failed");
    }
    vkBindImageMemory(device, storage_image, storage_memory, 0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = storage_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &vci, nullptr, &storage_view) != VK_SUCCESS) {
      throw std::runtime_error("storage image view failed");
    }
  }

  VkShaderModule load_shader(const std::string& path) {
    const auto bytes = read_file(path);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateShaderModule failed for " + path);
    }
    return module;
  }

  void create_storage_and_pipeline() {
    create_storage_image();

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 1;
    dlci.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(device, &dlci, nullptr, &set_layout) != VK_SUCCESS) {
      throw std::runtime_error("descriptor set layout failed");
    }

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(FillPush);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout) != VK_SUCCESS) {
      throw std::runtime_error("pipeline layout failed");
    }

    const std::string spv = std::string(HWODS_SHADER_DIR) + "/fill.comp.spv";
    VkShaderModule module = load_shader(spv);
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &compute_pipeline) !=
        VK_SUCCESS) {
      throw std::runtime_error("compute pipeline failed");
    }
    vkDestroyShaderModule(device, module, nullptr);

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    dpci.maxSets = 1;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &descriptor_pool) != VK_SUCCESS) {
      throw std::runtime_error("descriptor pool failed");
    }

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    if (vkAllocateDescriptorSets(device, &dsai, &descriptor_set) != VK_SUCCESS) {
      throw std::runtime_error("allocate descriptor set failed");
    }
    update_descriptors();
  }

  void update_descriptors() {
    VkDescriptorImageInfo img{};
    img.imageView = storage_view;
    img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor_set;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &img;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }

  void create_frames() {
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &cpci, nullptr, &command_pool) != VK_SUCCESS) {
      throw std::runtime_error("command pool failed");
    }

    const uint32_t frames = static_cast<uint32_t>(swapchain_images.size());
    command_buffers.resize(frames);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = command_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = frames;
    if (vkAllocateCommandBuffers(device, &cai, command_buffers.data()) != VK_SUCCESS) {
      throw std::runtime_error("allocate command buffers failed");
    }

    image_available.resize(frames);
    render_finished.resize(frames);
    in_flight.resize(frames);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < frames; ++i) {
      vkCreateSemaphore(device, &sci, nullptr, &image_available[i]);
      vkCreateSemaphore(device, &sci, nullptr, &render_finished[i]);
      vkCreateFence(device, &fci, nullptr, &in_flight[i]);
    }
  }

  void destroy_swapchain_resources() {
    if (!device) return;
    if (storage_view) vkDestroyImageView(device, storage_view, nullptr);
    if (storage_image) vkDestroyImage(device, storage_image, nullptr);
    if (storage_memory) vkFreeMemory(device, storage_memory, nullptr);
    storage_view = VK_NULL_HANDLE;
    storage_image = VK_NULL_HANDLE;
    storage_memory = VK_NULL_HANDLE;
    for (auto v : swapchain_views) vkDestroyImageView(device, v, nullptr);
    swapchain_views.clear();
    swapchain_images.clear();
    if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }

  void recreate_swapchain() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    while (w == 0 || h == 0) {
      glfwGetFramebufferSize(window, &w, &h);
      glfwWaitEvents();
    }
    vkDeviceWaitIdle(device);
    destroy_swapchain_resources();
    if (descriptor_pool) {
      vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      descriptor_pool = VK_NULL_HANDLE;
    }
    create_swapchain();
    create_storage_image();

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    dpci.maxSets = 1;
    vkCreateDescriptorPool(device, &dpci, nullptr, &descriptor_pool);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    vkAllocateDescriptorSets(device, &dsai, &descriptor_set);
    update_descriptors();
  }

  void draw_frame() {
    const std::size_t fi = frame_index % in_flight.size();
    vkWaitForFences(device, 1, &in_flight[fi], VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_available[fi],
                                         VK_NULL_HANDLE, &image_index);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR || framebuffer_resized) {
      framebuffer_resized = false;
      recreate_swapchain();
      return;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("acquire failed");
    }

    vkResetFences(device, 1, &in_flight[fi]);
    VkCommandBuffer cmd = command_buffers[fi];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin);

    image_barrier(cmd, storage_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    const auto elapsed = std::chrono::steady_clock::now() - start;
    FillPush push{};
    push.resolution[0] = static_cast<float>(extent.width);
    push.resolution[1] = static_cast<float>(extent.height);
    push.time = std::chrono::duration<float>(elapsed).count();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                            &descriptor_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                       &push);
    vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);

    image_barrier(cmd, storage_image, VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT);
    image_barrier(cmd, swapchain_images[image_index], VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = {static_cast<int32_t>(extent.width),
                            static_cast<int32_t>(extent.height), 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                            static_cast<int32_t>(extent.height), 1};
    vkCmdBlitImage(cmd, storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &region, VK_FILTER_NEAREST);

    image_barrier(cmd, swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &image_available[fi];
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &render_finished[fi];
    if (vkQueueSubmit(queue, 1, &submit, in_flight[fi]) != VK_SUCCESS) {
      throw std::runtime_error("queue submit failed");
    }

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &render_finished[fi];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &image_index;
    VkResult pr = vkQueuePresentKHR(queue, &present);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
      framebuffer_resized = false;
      recreate_swapchain();
    } else if (pr != VK_SUCCESS) {
      throw std::runtime_error("present failed");
    }

    frame_index++;
  }

  void loop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      draw_frame();
    }
    vkDeviceWaitIdle(device);
  }
};

VulkanFillApp::VulkanFillApp(int width, int height) : impl_(std::make_unique<Impl>(width, height)) {}
VulkanFillApp::~VulkanFillApp() = default;

void VulkanFillApp::run() { impl_->loop(); }

std::string vulkan_loader_hint() {
  return "Set VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json if devices are "
         "missing. Homebrew libvulkan: /opt/homebrew/lib/libvulkan.dylib";
}

}  // namespace hwods
