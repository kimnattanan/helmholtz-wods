#include "hwods/gpu/vulkan_fill.hpp"

#include <iostream>
#include <stdexcept>

int main() {
  std::cout << "helmholtz-wods — Walk on Decomposed Subdomains (C++)\n";
  std::cout << "Week 1: compute dispatch fill (Pass A/C/D come online W2–W4)\n";
  try {
    hwods::VulkanFillApp app;
    app.run();
  } catch (const std::exception& ex) {
    std::cerr << "Fatal: " << ex.what() << "\n";
    std::cerr << hwods::vulkan_loader_hint() << "\n";
    return 1;
  }
  return 0;
}
