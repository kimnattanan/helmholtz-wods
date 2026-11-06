//! Vulkan 1.3 (MoltenVK) path: Pass A/C compute, Pass D graphics, Pass B on CPU.
//! Week 1: winit window + compute fill of the swapchain.

mod fill_app;
mod vulkan;

pub use fill_app::run_fill_window;
pub use vulkan::VulkanContext;
