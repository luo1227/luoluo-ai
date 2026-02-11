mod inference;

use crate::inference::InferenceEngine;
use eframe::egui;
use rfd::FileDialog;
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

struct DetectionApp {
    engine: Arc<InferenceEngine>,
    model_path: String,
    device_type: String,
    status: String,
}

impl DetectionApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut fonts = egui::FontDefinitions::default();
        if let Ok(data) = std::fs::read("C:/Windows/Fonts/msyh.ttc") {
            fonts.font_data.insert(
                "zh_font".to_owned(),
                egui::FontData::from_owned(data),
            );
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                family.insert(0, "zh_font".to_owned());
            }
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                family.insert(0, "zh_font".to_owned());
            }
            cc.egui_ctx.set_fonts(fonts);
        }

        Self {
            engine: Arc::new(InferenceEngine::new()),
            model_path: "未选择".to_string(),
            device_type: "cpu".to_string(),
            status: "就绪".to_string(),
        }
    }
}

impl eframe::App for DetectionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("USLS 屏幕检测控制中心");
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("模型路径:");
                ui.label(&self.model_path);
                if ui.button("选择模型").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("ONNX Model", &["onnx"])
                        .pick_file() 
                    {
                        self.model_path = path.display().to_string();
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("推理后端:");
                ui.selectable_value(&mut self.device_type, "cpu".to_string(), "CPU");
                ui.selectable_value(&mut self.device_type, "cuda".to_string(), "CUDA");
                ui.selectable_value(&mut self.device_type, "tensorrt".to_string(), "TensorRT");
                ui.selectable_value(&mut self.device_type, "nvrtx".to_string(), "TensorRT-RTX");
            });

            let mut conf = *self.engine.conf_threshold.lock();
            if ui.add(egui::Slider::new(&mut conf, 0.0..=1.0).text("置信度阈值")).changed() {
                *self.engine.conf_threshold.lock() = conf;
            }

            ui.add_space(20.0);

            if ui.button("加载模型").clicked() {
                self.status = "正在加载...".to_string();
                let engine = self.engine.clone();
                let path = std::path::PathBuf::from(&self.model_path);
                let device = self.device_type.clone();
                
                std::thread::spawn(move || {
                    match engine.load_model(path, &device) {
                        Ok(_) => {
                            info!("模型加载成功");
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            if cfg!(debug_assertions) {
                                error!("模型加载失败: {}", msg);
                            } else if msg.contains("Please compile ONNXRuntime with CUDA") {
                                error!("模型加载失败: 当前 ONNX Runtime 未启用 CUDA，请安装带 CUDA 支持的版本或调整推理后端设置");
                            } else if msg.contains("Please compile ONNXRuntime with NVTensorRT-RTX") {
                                error!("模型加载失败: 当前 ONNX Runtime 未启用 TensorRT-RTX，请安装带 TensorRT-RTX 支持的版本或调整推理后端设置");
                            } else if msg.contains("Please compile ONNXRuntime with TensorRT") {
                                error!("模型加载失败: 当前 ONNX Runtime 未启用 TensorRT，请安装带 TensorRT 支持的版本或调整推理后端设置");
                            } else {
                                error!("模型加载失败: {}", msg);
                            }
                        }
                    }
                });
            }

            if ui.button("开始检测").clicked() {
                if let Err(e) = crate::inference::start_capture(self.engine.clone()) {
                    self.status = format!("启动失败: {}", e);
                } else {
                    self.status = "正在检测".to_string();
                    *self.engine.is_running.lock() = true;
                    // 启动 Overlay
                    start_overlay(self.engine.clone());
                }
            }

            ui.add_space(10.0);
            ui.label(format!("状态: {}", self.status));
        });
    }
}

struct OverlayApp {
    engine: Arc<InferenceEngine>,
}

impl eframe::App for OverlayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 全屏透明窗口，不显示任何 UI 组件，只绘制框
        let painter = ctx.layer_painter(egui::LayerId::background());
        
        let results = self.engine.results.lock();
        for y in results.iter() {
            for hbb in &y.hbbs {
                let rect = egui::Rect::from_min_max(
                    egui::pos2(hbb.xmin(), hbb.ymin()),
                    egui::pos2(hbb.xmax(), hbb.ymax()),
                );
                painter.rect_stroke(rect, 0.0, egui::Stroke::new(2.0, egui::Color32::RED));
            }
        }

        // 持续重绘
        ctx.request_repaint();
    }
}

fn start_overlay(engine: Arc<InferenceEngine>) {
    let _ = engine;
}

fn main() -> eframe::Result<()> {
    let default_filter = if cfg!(debug_assertions) { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "USLS 控制中心",
        options,
        Box::new(|cc| Box::new(DetectionApp::new(cc))),
    )
}
