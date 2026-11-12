#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct GLFWwindow;

namespace hwods {

/// Week-1: GLFW window + Vulkan compute fill of the swapchain.
class VulkanFillApp {
 public:
  explicit VulkanFillApp(int width = 1280, int height = 720);
  ~VulkanFillApp();

  VulkanFillApp(const VulkanFillApp&) = delete;
  VulkanFillApp& operator=(const VulkanFillApp&) = delete;

  /// Blocking event loop until the window closes. Throws std::runtime_error on fatal GPU errors.
  void run();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::string vulkan_loader_hint();

}  // namespace hwods
