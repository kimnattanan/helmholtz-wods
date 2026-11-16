#include "hwods/gpu/submit_timer.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hwods {

void GpuSubmitTimer::init(VkDevice device_in, VkPhysicalDevice physical, uint32_t queue_family,
                          uint32_t query_count) {
  device = device_in;
  timestamps = false;
  pool = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical, &props);
  period_ns = props.limits.timestampPeriod;

  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  if (family_count > 0) {
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
  }
  if (queue_family < family_count) valid_bits = families[queue_family].timestampValidBits;

  const bool supported = props.limits.timestampComputeAndGraphics == VK_TRUE && valid_bits > 0 &&
                         period_ns > 0.f && query_count >= 2;
  if (!supported) return;

  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = query_count;
  if (vkCreateQueryPool(device, &info, nullptr, &pool) != VK_SUCCESS) {
    pool = VK_NULL_HANDLE;
    return;
  }
  timestamps = true;
}

void GpuSubmitTimer::destroy() {
  if (device && pool) vkDestroyQueryPool(device, pool, nullptr);
  pool = VK_NULL_HANDLE;
  timestamps = false;
  device = VK_NULL_HANDLE;
}

void GpuSubmitTimer::mark_begin(VkCommandBuffer cmd, uint32_t first_query) const {
  if (!timestamps) return;
  vkCmdResetQueryPool(cmd, pool, first_query, 2);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, pool, first_query);
}

void GpuSubmitTimer::mark_end(VkCommandBuffer cmd, uint32_t first_query) const {
  if (!timestamps) return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, pool, first_query + 1);
}

std::optional<double> GpuSubmitTimer::gpu_milliseconds(uint32_t first_query) {
  if (!timestamps || !pool) return std::nullopt;

  uint64_t stamps[2]{};
  const VkResult result = vkGetQueryPoolResults(
      device, pool, first_query, 2, sizeof(stamps), stamps, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (result == VK_ERROR_DEVICE_LOST) {
    throw std::runtime_error("timestamp query lost the device");
  }
  if (result != VK_SUCCESS || (stamps[0] == 0 && stamps[1] == 0)) {
    timestamps = false;
    return std::nullopt;
  }

  const uint64_t mask = valid_bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << valid_bits) - 1;
  const uint64_t start = stamps[0] & mask;
  const uint64_t end = stamps[1] & mask;
  const uint64_t delta = end >= start ? end - start : (mask - start) + 1 + end;
  return static_cast<double>(delta) * static_cast<double>(period_ns) / 1.0e6;
}

void GpuSubmitTimer::report(const char* pass, uint32_t first_query,
                            std::chrono::steady_clock::time_point submit_time) {
  bool used_fence = false;
  double ms = 0.0;
  if (const std::optional<double> gpu = gpu_milliseconds(first_query)) {
    ms = *gpu;
  } else {
    used_fence = true;
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submit_time)
             .count();
    if (!logged_fallback) {
      std::cout << "MoltenVK rejected timestamp queries; timing submits with a fence\n";
      logged_fallback = true;
    }
  }

  std::ostringstream text;
  text << std::fixed << std::setprecision(3) << ms;
  std::cout << pass << ' ' << text.str() << " ms";
  if (used_fence) std::cout << " (fence)";
  std::cout << std::endl;
}

}  // namespace hwods
