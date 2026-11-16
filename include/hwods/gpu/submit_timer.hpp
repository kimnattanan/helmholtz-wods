#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <optional>

namespace hwods {

/// GPU timestamp around one compute submit.
/// Pass B stays a CPU fence later; this does not time an I−Q solve.
class GpuSubmitTimer {
 public:
  void init(VkDevice device, VkPhysicalDevice physical, uint32_t queue_family,
            uint32_t query_count);
  void destroy();

  void mark_begin(VkCommandBuffer cmd, uint32_t first_query) const;
  void mark_end(VkCommandBuffer cmd, uint32_t first_query) const;

  /// After the submit fence has signaled. Prints milliseconds.
  /// Falls back to the CPU interval since `submit_time` when timestamp queries are rejected.
  void report(const char* pass, uint32_t first_query,
              std::chrono::steady_clock::time_point submit_time);

 private:
  std::optional<double> gpu_milliseconds(uint32_t first_query);

  VkDevice device = VK_NULL_HANDLE;
  VkQueryPool pool = VK_NULL_HANDLE;
  float period_ns = 0.f;
  uint32_t valid_bits = 0;
  bool timestamps = false;
  bool logged_fallback = false;
};

}  // namespace hwods
