mod capture;
mod inference;

use crate::inference::{CaptureConfig, InferenceEngine, RegionMode};
use eframe::egui;
use parking_lot::Mutex;
use rfd::FileDialog;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogLevel {
    Debug,
    Info,
    Error,
}

struct LogEntry {
    level: LogLevel,
    message: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GuiRegionMode {
    Fullscreen,
    Center640,
    Center1280,
    Custom,
}

struct DetectionApp {
    engine: Arc<InferenceEngine>,
    model_path: String,
    device_type: String,
    yolo_version: u8,
    gui_region_mode: GuiRegionMode,
    custom_x: u32,
    custom_y: u32,
    custom_width: u32,
    custom_height: u32,
    selected_tab: usize,
    logs: Arc<Mutex<Vec<LogEntry>>>,
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
            yolo_version: 26,
            gui_region_mode: GuiRegionMode::Fullscreen,
            custom_x: 0,
            custom_y: 0,
            custom_width: 1280,
            custom_height: 720,
            selected_tab: 0,
            logs: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl eframe::App for DetectionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            let is_running = *self.engine.is_running.lock();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!is_running, egui::Button::new("选择模型并加载"))
                    .clicked()
                {
                    let Some(path) = FileDialog::new()
                        .add_filter("ONNX Model", &["onnx"])
                        .pick_file()
                    else {
                        return;
                    };
                    self.model_path = path.display().to_string();
                    let device = self.device_type.clone();
                    let yolo_version = self.yolo_version;

                    match self
                        .engine
                        .load_model(path, &device, Some(yolo_version))
                    {
                        Ok(_) => {
                            info!("模型加载成功");
                            push_log(&self.logs, LogLevel::Info, "模型加载成功");
                        }
                        Err(e) => {
                            let msg = format!("{:#}", e);
                            push_log(&self.logs, LogLevel::Error, format!("模型加载失败: {}", msg));
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
                }

                if ui.add_enabled(!is_running, egui::Button::new("开始检测")).clicked() {
                    *self.engine.is_running.lock() = true;
                    push_log(&self.logs, LogLevel::Info, "开始检测");
                    let engine = self.engine.clone();
                    let logs = self.logs.clone();
                    thread::spawn(move || {
                        let mut ctx = match crate::capture::create_capture_context(0) {
                            Ok(ctx) => ctx,
                            Err(e) => {
                                push_log(&logs, LogLevel::Error, format!("捕获初始化失败: {}", e));
                                *engine.is_running.lock() = false;
                                return;
                            }
                        };
                        let mut last_report = Instant::now();
                        let mut frames: u64 = 0;
                        let mut capture_total = 0.0;
                        let mut infer_total = 0.0;
                        loop {
                            if !*engine.is_running.lock() {
                                break;
                            }
                            match engine.capture_once(&mut ctx) {
                                Ok(stats) => {
                                    if stats.did_capture {
                                        frames += 1;
                                        capture_total += stats.capture_ms;
                                        infer_total += stats.infer_ms;
                                    }
                                }
                                Err(e) => {
                                    push_log(&logs, LogLevel::Error, format!("捕获失败: {}", e));
                                    *engine.is_running.lock() = false;
                                    break;
                                }
                            }
                            if last_report.elapsed() >= Duration::from_secs(1) {
                                let elapsed = last_report.elapsed().as_secs_f64();
                                let (fps, cap_ms, infer_ms) = if frames > 0 {
                                    (
                                        frames as f64 / elapsed,
                                        capture_total / frames as f64,
                                        infer_total / frames as f64,
                                    )
                                } else {
                                    (0.0, 0.0, 0.0)
                                };
                                push_log(
                                    &logs,
                                    LogLevel::Info,
                                    format!("fps: {:.1} 捕获{:.2}ms 推理{:.2}ms", fps, cap_ms, infer_ms),
                                );
                                frames = 0;
                                capture_total = 0.0;
                                infer_total = 0.0;
                                last_report = Instant::now();
                            }
                            thread::sleep(Duration::from_millis(1));
                        }
                    });
                }

                if ui.add_enabled(is_running, egui::Button::new("结束检测")).clicked() {
                    *self.engine.is_running.lock() = false;
                    push_log(&self.logs, LogLevel::Info, "结束检测");
                }
            });
            ui.add_space(10.0);
            ui.label("日志输出:");
            let show_debug = cfg!(debug_assertions);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        let logs = self.logs.lock();
                        for entry in logs.iter() {
                            if entry.level == LogLevel::Debug && !show_debug {
                                continue;
                            }
                            let prefix = match entry.level {
                                LogLevel::Debug => "[DEBUG] ",
                                LogLevel::Info => "[INFO] ",
                                LogLevel::Error => "[ERROR] ",
                            };
                            ui.label(format!("{}{}", prefix, entry.message));
                        }
                    });
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("USLS 屏幕检测控制中心");
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.selected_tab, 0, "推理配置");
                ui.selectable_value(&mut self.selected_tab, 1, "捕获设置");
            });
            ui.add_space(10.0);
            if self.selected_tab == 0 {
                ui.horizontal(|ui| {
                    ui.label("推理后端:");
                    ui.selectable_value(&mut self.device_type, "cpu".to_string(), "CPU");
                    ui.selectable_value(&mut self.device_type, "cuda".to_string(), "CUDA");
                    ui.selectable_value(&mut self.device_type, "tensorrt".to_string(), "TensorRT");
                });

                ui.horizontal(|ui| {
                    ui.label("模型性能优化:");
                    ui.label("(开启 CUDA 后建议使用 TensorRT 或 FP16 模型)");
                });

                ui.horizontal(|ui| {
                    ui.label("YOLO版本:");
                    ui.selectable_value(&mut self.yolo_version, 26, "26");
                    ui.selectable_value(&mut self.yolo_version, 13, "13");
                    ui.selectable_value(&mut self.yolo_version, 12, "12");
                    ui.selectable_value(&mut self.yolo_version, 11, "11");
                    ui.selectable_value(&mut self.yolo_version, 10, "10");
                    ui.selectable_value(&mut self.yolo_version, 9, "9");
                    ui.selectable_value(&mut self.yolo_version, 8, "8");
                    ui.selectable_value(&mut self.yolo_version, 7, "7");
                    ui.selectable_value(&mut self.yolo_version, 6, "6");
                    ui.selectable_value(&mut self.yolo_version, 5, "5");
                });

                ui.horizontal(|ui| {
                    ui.label("置信度阈值:");
                    let mut conf = *self.engine.conf_threshold.lock();
                    if ui.add(egui::Slider::new(&mut conf, 0.0..=1.0)).changed() {
                        *self.engine.conf_threshold.lock() = conf;
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    ui.label("区域设置:");
                    ui.radio_value(&mut self.gui_region_mode, GuiRegionMode::Center640, "中心 640*640");
                    ui.radio_value(&mut self.gui_region_mode, GuiRegionMode::Center1280, "中心 1280*1280");
                    ui.radio_value(&mut self.gui_region_mode, GuiRegionMode::Fullscreen, "全屏");
                    ui.radio_value(&mut self.gui_region_mode, GuiRegionMode::Custom, "自定义");
                });

                if self.gui_region_mode == GuiRegionMode::Custom {
                    ui.horizontal(|ui| {
                        ui.label("屏幕左上角坐标x:");
                        ui.add(egui::DragValue::new(&mut self.custom_x).speed(1));
                        ui.label("屏幕左上角坐标y:");
                        ui.add(egui::DragValue::new(&mut self.custom_y).speed(1));
                    });
                    ui.horizontal(|ui| {
                        ui.label("宽度:");
                        ui.add(egui::DragValue::new(&mut self.custom_width).speed(1));
                        ui.label("高度:");
                        ui.add(egui::DragValue::new(&mut self.custom_height).speed(1));
                    });
                }

                let mode = match self.gui_region_mode {
                    GuiRegionMode::Fullscreen => RegionMode::Fullscreen,
                    GuiRegionMode::Center640 => RegionMode::Center640,
                    GuiRegionMode::Center1280 => RegionMode::Center1280,
                    GuiRegionMode::Custom => RegionMode::Custom {
                        x: self.custom_x,
                        y: self.custom_y,
                        width: self.custom_width,
                        height: self.custom_height,
                    },
                };

                let mut config = self.engine.capture_config.lock();
                *config = CaptureConfig { mode };
            }
        });
        ctx.request_repaint();
    }
}

fn push_log(logs: &Arc<Mutex<Vec<LogEntry>>>, level: LogLevel, message: impl Into<String>) {
    let mut logs = logs.lock();
    logs.push(LogEntry {
        level,
        message: message.into(),
    });
    if logs.len() > 500 {
        let overflow = logs.len() - 500;
        logs.drain(0..overflow);
    }
}

fn main() -> eframe::Result<()> {
    let default_filter = if cfg!(debug_assertions) {
        "debug"
    } else {
        "info,warn,error"
    };
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
