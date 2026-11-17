#include "hwods/gpu/vulkan_pass_a.hpp"
#include "hwods/algo/sdf.hpp"
#include "hwods/gpu/submit_timer.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hwods {
namespace {

// std140 Frame. Offsets match pass_a.comp SPIR-V (tank_size 0, k 8, … node_count 28).
struct GpuFrame {
  float tank_size[2];
  float k;
  float eps;
  uint32_t max_steps;
  uint32_t walks_per_bucket;
  uint32_t seed;
  uint32_t node_count;
};

// std430 SdfNode. Array stride 48; mat3x2 is column-major with stride 8.
struct GpuSdfNode {
  uint32_t kind;
  float radius;
  float half_extents[2];
  float transform[3][2];
  uint32_t op_with_next;
  uint32_t pad;
};

static_assert(sizeof(GpuFrame) == 32, "Frame uniform must be 32 bytes");
static_assert(offsetof(GpuFrame, k) == 8, "Frame.k offset");
static_assert(offsetof(GpuFrame, eps) == 12, "Frame.eps offset");
static_assert(offsetof(GpuFrame, max_steps) == 16, "Frame.max_steps offset");
static_assert(offsetof(GpuFrame, node_count) == 28, "Frame.node_count offset");
static_assert(sizeof(GpuSdfNode) == 48, "SdfNode SSBO stride must be 48");
static_assert(offsetof(GpuSdfNode, half_extents) == 8, "SdfNode.half_extents offset");
static_assert(offsetof(GpuSdfNode, transform) == 16, "SdfNode.transform offset");
static_assert(offsetof(GpuSdfNode, op_with_next) == 40, "SdfNode.op_with_next offset");
static_assert(sizeof(PassABin) == 8, "histogram vec2 stride must be 8");

constexpr uint32_t kLocalSizeX = 64;

void vk_check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) throw std::runtime_error(what);
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file) throw std::runtime_error("failed to read " + path);
  const auto size = static_cast<std::size_t>(file.tellg());
  std::vector<char> bytes(size);
  file.seekg(0);
  file.read(bytes.data(), static_cast<std::streamsize>(size));
  return bytes;
}

uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t type_bits,
                          VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(physical, &mem);
  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  throw std::runtime_error("no matching memory type");
}

GpuSdfNode pack_node(const SdfNode& node) {
  GpuSdfNode gpu{};
  gpu.kind = static_cast<uint32_t>(node.kind);
  gpu.radius = node.radius;
  gpu.half_extents[0] = node.half_extents[0];
  gpu.half_extents[1] = node.half_extents[1];
  for (int column = 0; column < 3; ++column) {
    gpu.transform[column][0] = node.transform[static_cast<std::size_t>(column)][0];
    gpu.transform[column][1] = node.transform[static_cast<std::size_t>(column)][1];
  }
  gpu.op_with_next = static_cast<uint32_t>(node.op_with_next);
  gpu.pad = 0;
  return gpu;
}

}  // namespace

struct VulkanPassA::Impl {
  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
  };

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;

  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  GpuSubmitTimer timer;

  Buffer frame_buffer;
  Buffer node_buffer;
  Buffer histogram_buffer;

  Impl() {
    create_instance();
    pick_device();
    create_device();
    create_pipeline();
    create_commands();
    timer.init(device, physical, queue_family, 2);
  }

  ~Impl() {
    if (device) vkDeviceWaitIdle(device);
    destroy_buffer(frame_buffer);
    destroy_buffer(node_buffer);
    destroy_buffer(histogram_buffer);
    if (device) {
      timer.destroy();
      if (fence) vkDestroyFence(device, fence, nullptr);
      if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
      if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
      if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
  }

  void create_instance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "helmholtz-wods";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "helmholtz-wods";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    const char* extensions[] = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };

    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = 2;
    info.ppEnabledExtensionNames = extensions;
    info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    vk_check(vkCreateInstance(&info, nullptr, &instance),
             "vkCreateInstance failed — check MoltenVK / libvulkan");
  }

  void pick_device() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("no Vulkan physical devices (enable portability)");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (VkPhysicalDevice candidate : devices) {
      uint32_t family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
      for (uint32_t i = 0; i < family_count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          physical = candidate;
          queue_family = i;
          return;
        }
      }
    }
    throw std::runtime_error("no compute queue");
  }

  void create_device() {
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, exts.data());
    const bool portability = std::any_of(
        exts.begin(), exts.end(), [](const VkExtensionProperties& ext) {
          return std::strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0;
        });

    const char* portability_name = "VK_KHR_portability_subset";
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    if (portability) {
      info.enabledExtensionCount = 1;
      info.ppEnabledExtensionNames = &portability_name;
    }
    vk_check(vkCreateDevice(physical, &info, nullptr, &device), "vkCreateDevice failed");
    vkGetDeviceQueue(device, queue_family, 0, &queue);
  }

  VkShaderModule load_shader(const std::string& path) {
    const auto bytes = read_file(path);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes.size();
    info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule module = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(device, &info, nullptr, &module),
             ("vkCreateShaderModule failed for " + path).c_str());
    return module;
  }

  void create_pipeline() {
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;
    vk_check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout),
             "descriptor set layout failed");

    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    vk_check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
             "pipeline layout failed");

    const std::string spv = std::string(HWODS_SHADER_DIR) + "/pass_a.comp.spv";
    VkShaderModule module = load_shader(spv);
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage;
    pipeline_info.layout = pipeline_layout;
    vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
             "Pass A compute pipeline failed");
    vkDestroyShaderModule(device, module, nullptr);

    VkDescriptorPoolSize pool_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    vk_check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
             "descriptor pool failed");

    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = descriptor_pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &set_layout;
    vk_check(vkAllocateDescriptorSets(device, &alloc, &descriptor_set),
             "allocate descriptor set failed");
  }

  void create_commands() {
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk_check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool), "command pool failed");

    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = command_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(device, &alloc, &command_buffer),
             "allocate command buffer failed");

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vk_check(vkCreateFence(device, &fence_info, nullptr, &fence), "fence failed");
  }

  Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &info, nullptr, &buffer.buffer), "vkCreateBuffer failed");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buffer.buffer, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(
        physical, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vk_check(vkAllocateMemory(device, &alloc, nullptr, &buffer.memory), "vkAllocateMemory failed");
    vk_check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory failed");
    vk_check(vkMapMemory(device, buffer.memory, 0, req.size, 0, &buffer.mapped), "vkMapMemory failed");
    return buffer;
  }

  void destroy_buffer(Buffer& buffer) {
    if (!device) return;
    if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
    if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
  }

  void update_descriptors() {
    VkDescriptorBufferInfo frame_info{frame_buffer.buffer, 0, sizeof(GpuFrame)};
    VkDescriptorBufferInfo node_info{node_buffer.buffer, 0, node_buffer.size};
    VkDescriptorBufferInfo hist_info{histogram_buffer.buffer, 0, histogram_buffer.size};

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &frame_info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &node_info;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &hist_info;
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
  }

  std::vector<PassABin> dispatch(const PassAParams& params, const std::vector<SdfNode>& nodes,
                                 uint32_t histogram_bins) {
    if (histogram_bins == 0) throw std::runtime_error("Pass A histogram needs at least one bin");

    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    destroy_buffer(frame_buffer);
    destroy_buffer(node_buffer);
    destroy_buffer(histogram_buffer);

    frame_buffer = create_buffer(sizeof(GpuFrame), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const VkDeviceSize node_bytes =
        sizeof(GpuSdfNode) * std::max<std::size_t>(nodes.size(), 1);
    node_buffer = create_buffer(node_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    histogram_buffer =
        create_buffer(sizeof(PassABin) * histogram_bins, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    GpuFrame frame{};
    frame.tank_size[0] = params.tank_size[0];
    frame.tank_size[1] = params.tank_size[1];
    frame.k = params.k;
    frame.eps = params.eps;
    frame.max_steps = params.max_steps;
    frame.walks_per_bucket = params.walks_per_bucket;
    frame.seed = params.seed;
    frame.node_count = static_cast<uint32_t>(nodes.size());
    std::memcpy(frame_buffer.mapped, &frame, sizeof(frame));

    std::vector<GpuSdfNode> packed(std::max<std::size_t>(nodes.size(), 1));
    for (std::size_t i = 0; i < nodes.size(); ++i) packed[i] = pack_node(nodes[i]);
    std::memcpy(node_buffer.mapped, packed.data(), sizeof(GpuSdfNode) * packed.size());
    std::memset(histogram_buffer.mapped, 0, static_cast<std::size_t>(histogram_buffer.size));

    update_descriptors();

    vkResetCommandBuffer(command_buffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(command_buffer, &begin), "begin command buffer failed");

    VkBufferMemoryBarrier upload[3]{};
    for (int i = 0; i < 3; ++i) {
      upload[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      upload[i].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      upload[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      upload[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      upload[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    upload[0].buffer = frame_buffer.buffer;
    upload[0].size = frame_buffer.size;
    upload[1].buffer = node_buffer.buffer;
    upload[1].size = node_buffer.size;
    upload[2].buffer = histogram_buffer.buffer;
    upload[2].size = histogram_buffer.size;
    upload[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 3, upload, 0,
                         nullptr);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                            &descriptor_set, 0, nullptr);
    const uint32_t groups = (histogram_bins + kLocalSizeX - 1) / kLocalSizeX;
    timer.mark_begin(command_buffer, 0);
    vkCmdDispatch(command_buffer, groups, 1, 1);
    timer.mark_end(command_buffer, 0);

    VkBufferMemoryBarrier readback{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    readback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    readback.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback.buffer = histogram_buffer.buffer;
    readback.size = histogram_buffer.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &readback, 0, nullptr);

    vk_check(vkEndCommandBuffer(command_buffer), "end command buffer failed");

    vkResetFences(device, 1, &fence);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    const auto submitted = std::chrono::steady_clock::now();
    vk_check(vkQueueSubmit(queue, 1, &submit, fence), "Pass A submit failed");
    vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "Pass A fence wait failed");
    timer.report("pass-a", 0, submitted);

    std::vector<PassABin> bins(histogram_bins);
    std::memcpy(bins.data(), histogram_buffer.mapped, sizeof(PassABin) * histogram_bins);
    return bins;
  }
};

VulkanPassA::VulkanPassA() : impl_(std::make_unique<Impl>()) {}
VulkanPassA::~VulkanPassA() = default;

std::vector<PassABin> VulkanPassA::dispatch(const PassAParams& params,
                                            const std::vector<SdfNode>& nodes,
                                            uint32_t histogram_bins) {
  return impl_->dispatch(params, nodes, histogram_bins);
}

std::vector<SdfNode> demo_pass_a_nodes() {
  SdfNode circle = SdfNode::circle({0.35f, 0.50f}, 0.12f);
  circle.op_with_next = SdfOp::Union;

  SdfNode box = SdfNode::box_centered({0.68f, 0.42f}, {0.10f, 0.07f});
  box.op_with_next = SdfOp::Difference;

  SdfNode rounded;
  rounded.kind = SdfPrimitiveKind::RoundedBox;
  rounded.radius = 0.02f;
  rounded.half_extents = {0.08f, 0.05f};
  rounded.transform[2] = {0.50f, 0.22f};
  rounded.op_with_next = SdfOp::Union;
  return {circle, box, rounded};
}

PassABin run_pass_a_demo() {
  PassAParams params;
  params.tank_size = {1.f, 1.f};
  params.k = 0.f;
  params.eps = 1e-3f;
  params.max_steps = 32;
  params.walks_per_bucket = 64;
  params.seed = 1;

  SceneSdf scene = make_example_scene(params.tank_size[0], params.tank_size[1]);

  VulkanPassA pass;
  const std::vector<PassABin> bins = pass.dispatch(params, scene.obstacles, 1);
  if (bins.size() != 1) throw std::runtime_error("Pass A returned no histogram bin");

  const std::array<float, 2> center{params.tank_size[0] * 0.5f, params.tank_size[1] * 0.5f};
  const float expected_distance = scene.domain_distance(center);
  const float expected_count = static_cast<float>(scene.obstacles.size());
  if (std::fabs(bins[0].re - expected_distance) > 1e-4f ||
      std::fabs(bins[0].im - expected_count) > 1e-4f) {
    throw std::runtime_error(
        "Pass A histogram disagrees with CPU SDF (got " + std::to_string(bins[0].re) + ", " +
        std::to_string(bins[0].im) + " expected " + std::to_string(expected_distance) + ", " +
        std::to_string(expected_count) + ")");
  }
  return bins[0];
}

}  // namespace hwods
