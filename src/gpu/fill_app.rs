//! Week-1 demo: winit window that dispatches compute and fills the screen.

use crate::gpu::vulkan::VulkanContext;
use anyhow::Result;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct FillApp {
    window: Option<Arc<Window>>,
    vulkan: Option<VulkanContext>,
}

impl ApplicationHandler for FillApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("helmholtz-wods — Week 1 compute fill")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        match event_loop.create_window(attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                match VulkanContext::new(window.as_ref()) {
                    Ok(vk) => {
                        self.vulkan = Some(vk);
                        self.window = Some(window);
                        event_loop.set_control_flow(ControlFlow::Poll);
                    }
                    Err(err) => {
                        eprintln!("Vulkan init failed: {err:?}");
                        event_loop.exit();
                    }
                }
            }
            Err(err) => {
                eprintln!("window create failed: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(vk) = self.vulkan.as_mut() {
                    if let Err(err) = vk.resize(size.width, size.height) {
                        eprintln!("resize failed: {err:?}");
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(vk) = self.vulkan.as_mut() {
                    if let Err(err) = vk.draw_frame() {
                        eprintln!("draw_frame failed: {err:?}");
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

pub fn run_fill_window() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = FillApp {
        window: None,
        vulkan: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
