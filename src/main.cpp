#include "hwods/gpu/vulkan_fill.hpp"
#include "hwods/gpu/vulkan_pass_a.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  std::cout << "helmholtz-wods — Walk on Decomposed Subdomains (C++)\n";
  try {
    if (argc > 1 && std::string(argv[1]) == "--pass-a") {
      const hwods::PassABin bin = hwods::run_pass_a_demo();
      std::cout << "Pass A weights[0] = (" << bin.re << ", " << bin.im << ")\n";
      return 0;
    }
    std::cout << "Week 1: compute dispatch fill (Pass A/C/D come online W2–W4)\n";
    hwods::VulkanFillApp app;
    app.run();
  } catch (const std::exception& ex) {
    std::cerr << "Fatal: " << ex.what() << "\n";
    std::cerr << hwods::vulkan_loader_hint() << "\n";
    return 1;
  }
  return 0;
}
