mod capture;
mod config;
mod inference;
mod input_listener;
mod mouse_control;

use crate::capture::ScreenCapture;
use crate::config::ConfigManager;
use crate::inference::YoloInferencer;
use crate::input_listener::InputListener;
use eframe::egui;
use parking_lot::Mutex;
use rfd::FileDialog;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

#[derive(Clone, Copy)]
struct CaptureConfig {
    mode: GuiRegionMode,
}

struct DetectionApp {
    inferencer: Arc<YoloInferencer>,
    input_listener: Arc<InputListener>,
    is_running: Arc<AtomicBool>,
    capture_config: Arc<Mutex<CaptureConfig>>,
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
    config_manager: Arc<Mutex<Option<ConfigManager>>>,
    control_yaw_sensitivity: f32,
    control_pitch_sensitivity: f32,
    control_hotkey: String,
    control_trigger_toggle: bool,
    control_x_target_offset: f32,
    control_y_target_offset: f32,
    capturing_hotkey: bool,
}

impl DetectionApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut fonts = egui::FontDefinitions::default();
        if let Ok(data) = std::fs::read("C:/Windows/Fonts/msyh.ttc") {
            fonts
                .font_data
                .insert("zh_font".to_owned(), egui::FontData::from_owned(data));
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                family.insert(0, "zh_font".to_owned());
            }
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                family.insert(0, "zh_font".to_owned());
            }
            cc.egui_ctx.set_fonts(fonts);
        }

        let config_manager = ConfigManager::new(std::path::PathBuf::from("config.json"));
        let config = config_manager.load();

        Self {
            inferencer: Arc::new(YoloInferencer::new()),
            input_listener: Arc::new(InputListener::new()),
            is_running: Arc::new(AtomicBool::new(false)),
            capture_config: Arc::new(Mutex::new(CaptureConfig {
                mode: GuiRegionMode::Fullscreen,
            })),
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
            config_manager: Arc::new(Mutex::new(Some(config_manager))),
            control_yaw_sensitivity: config.control.yaw_sensitivity,
            control_pitch_sensitivity: config.control.pitch_sensitivity,
            control_hotkey: config.control.hotkey.clone(),
            control_trigger_toggle: config.control.trigger_type
                == crate::config::TriggerType::Toggle,
            control_x_target_offset: config.control.x_target_offset,
            control_y_target_offset: config.control.y_target_offset,
            capturing_hotkey: false,
        }
    }
}

impl eframe::App for DetectionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            let is_running = self.is_running.load(Ordering::SeqCst);
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
                        .inferencer
                        .load_model(&path, &device, yolo_version)
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
                    self.is_running.store(true, Ordering::SeqCst);
                    push_log(&self.logs, LogLevel::Info, "开始检测");
                    let inferencer = self.inferencer.clone();
                    let is_running = self.is_running.clone();
                    let capture_config = self.capture_config.clone();
                    let logs = self.logs.clone();
                    let trigger_toggle = self.control_trigger_toggle;
                    let custom_x = self.custom_x;
                    let custom_y = self.custom_y;
                    let custom_width = self.custom_width;
                    let custom_height = self.custom_height;
                    let control_config = crate::config::ControlSettings {
                        yaw_sensitivity: self.control_yaw_sensitivity,
                        pitch_sensitivity: self.control_pitch_sensitivity,
                        hotkey: self.control_hotkey.clone(),
                        trigger_type: if trigger_toggle {
                            crate::config::TriggerType::Toggle
                        } else {
                            crate::config::TriggerType::Hold
                        },
                        x_target_offset: self.control_x_target_offset,
                        y_target_offset: self.control_y_target_offset,
                    };
                    let input_listener = self.input_listener.clone();
                    input_listener.set_hotkey(&control_config.hotkey);
                    input_listener.set_toggle_state(false);
                    input_listener.start();
                    
                    thread::spawn(move || {
                        let mut capture = match ScreenCapture::try_new(0) {
                            Ok(c) => c,
                            Err(e) => {
                                push_log(&logs, LogLevel::Error, format!("捕获初始化失败: {}", e));
                                is_running.store(false, Ordering::SeqCst);
                                input_listener.stop();
                                return;
                            }
                        };
                        let mut last_report = Instant::now();
                        let mut frames: u64 = 0;
                        let mut capture_total = 0.0;
                        let mut infer_total = 0.0;
                        
                        let screen_center_x = capture.width() as f32 / 2.0;
                        let screen_center_y = capture.height() as f32 / 2.0;
                        
                        loop {
                            if !is_running.load(Ordering::SeqCst) {
                                break;
                            }
                            
                            let capture_start = Instant::now();
                            let config = *capture_config.lock();
                            let full_w = capture.width();
                            let full_h = capture.height();
                            
                            let (did_capture, width, height) = match config.mode {
                                GuiRegionMode::Fullscreen => {
                                    let did = capture.capture_full().map_err(|e| e.to_string());
                                    match did {
                                        Ok(d) => (d, full_w, full_h),
                                        Err(e) => {
                                            push_log(&logs, LogLevel::Error, format!("捕获失败: {}", e));
                                            is_running.store(false, Ordering::SeqCst);
                                            break;
                                        }
                                    }
                                }
                                GuiRegionMode::Center640 => {
                                    let w = 640.min(full_w);
                                    let h = 640.min(full_h);
                                    let x = (full_w - w) / 2;
                                    let y = (full_h - h) / 2;
                                    match capture.capture_region(x, y, w, h) {
                                        Ok(d) => (d, w, h),
                                        Err(e) => {
                                            push_log(&logs, LogLevel::Error, format!("捕获失败: {}", e));
                                            is_running.store(false, Ordering::SeqCst);
                                            break;
                                        }
                                    }
                                }
                                GuiRegionMode::Center1280 => {
                                    let w = 1280.min(full_w);
                                    let h = 1280.min(full_h);
                                    let x = (full_w - w) / 2;
                                    let y = (full_h - h) / 2;
                                    match capture.capture_region(x, y, w, h) {
                                        Ok(d) => (d, w, h),
                                        Err(e) => {
                                            push_log(&logs, LogLevel::Error, format!("捕获失败: {}", e));
                                            is_running.store(false, Ordering::SeqCst);
                                            break;
                                        }
                                    }
                                }
                                GuiRegionMode::Custom => {
                                    let (x, y, width, height) = (custom_x, custom_y, custom_width, custom_height);
                                    if width > 0 && height > 0 && x < full_w && y < full_h {
                                        let w = width.min(full_w - x);
                                        let h = height.min(full_h - y);
                                        match capture.capture_region(x, y, w, h) {
                                            Ok(d) => (d, w, h),
                                            Err(e) => {
                                                push_log(&logs, LogLevel::Error, format!("捕获失败: {}", e));
                                                is_running.store(false, Ordering::SeqCst);
                                                break;
                                            }
                                        }
                                    } else {
                                        match capture.capture_full() {
                                            Ok(d) => (d, full_w, full_h),
                                            Err(e) => {
                                                push_log(&logs, LogLevel::Error, format!("捕获失败: {}", e));
                                                is_running.store(false, Ordering::SeqCst);
                                                break;
                                            }
                                        }
                                    }
                                }
                            };
                            
                            let capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;
                            
                            if !did_capture || capture.rgba_buffer.is_empty() {
                                continue;
                            }
                            
                            let infer_start = Instant::now();
                            let infer_result = inferencer.infer_with_preprocess(&capture.rgba_buffer, width, height);
                            let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
                            
                            match infer_result {
                                Ok(results) => {
                                    frames += 1;
                                    capture_total += capture_ms;
                                    infer_total += infer_ms;
                                    
                                    if input_listener.is_hotkey_pressed() && input_listener.check_trigger_cooldown(100)
                                        && let Some(y) = results.first()
                                    {
                                        let boxes = y.hbbs();
                                        if !boxes.is_empty() {
                                                let mut closest_box: Option<[f32; 4]> = None;
                                                let mut min_dist = f32::MAX;
                                                
                                                for det in boxes {
                                                    let x1 = det.xmin();
                                                    let y1 = det.ymin();
                                                    let x2 = det.xmax();
                                                    let y2 = det.ymax();
                                                    let center_x = (x1 + x2) / 2.0;
                                                    let center_y = (y1 + y2) / 2.0;
                                                    let dist = ((center_x - screen_center_x).powi(2) + (center_y - screen_center_y).powi(2)).sqrt();
                                                    if dist < min_dist {
                                                        min_dist = dist;
                                                        closest_box = Some([x1, y1, x2, y2]);
                                                    }
                                                }
                                                
                                                if let Some(box_coords) = closest_box {
                                                    let x1 = box_coords[0];
                                                    let y1 = box_coords[1];
                                                    let x2 = box_coords[2];
                                                    let y2 = box_coords[3];
                                                    
                                                    let target_x = x1 + (x2 - x1) * (0.5 + control_config.x_target_offset);
                                                    let target_y = y1 + (y2 - y1) * (0.5 + control_config.y_target_offset);
                                                    
                                                    let dx = target_x - screen_center_x;
                                                    let dy = target_y - screen_center_y;
                                                    
                                                    let move_x = dx * control_config.yaw_sensitivity;
                                                    let move_y = dy * control_config.pitch_sensitivity;
                                                    
                                                    crate::mouse_control::move_relative(move_x as i32, move_y as i32);
                                                    
                                                    push_log(&logs, LogLevel::Debug, format!("移动: ({:.1}, {:.1}) -> ({:.1}, {:.1})", dx, dy, move_x, move_y));
                                                }
                                            }
                                        }
                                }
                                Err(e) => {
                                    push_log(&logs, LogLevel::Error, format!("推理失败: {}", e));
                                    is_running.store(false, Ordering::SeqCst);
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
                        input_listener.stop();
                    });
                }

                if ui.add_enabled(is_running, egui::Button::new("结束检测")).clicked() {
                    self.is_running.store(false, Ordering::SeqCst);
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
                ui.selectable_value(&mut self.selected_tab, 0, "控制设置");
                ui.selectable_value(&mut self.selected_tab, 1, "推理配置");
                ui.selectable_value(&mut self.selected_tab, 2, "捕获设置");
            });
            ui.add_space(10.0);
            if self.selected_tab == 0 {
                ui.horizontal(|ui| {
                    ui.label("Yaw灵敏度:");
                    ui.add(egui::Slider::new(
                        &mut self.control_yaw_sensitivity,
                        0.01..=2.0,
                    ));
                });

                ui.horizontal(|ui| {
                    ui.label("Pitch灵敏度:");
                    ui.add(egui::Slider::new(
                        &mut self.control_pitch_sensitivity,
                        0.01..=2.0,
                    ));
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("触发类型:");
                    ui.selectable_value(&mut self.control_trigger_toggle, false, "按下生效");
                    ui.selectable_value(&mut self.control_trigger_toggle, true, "按下切换");
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("热键:");
                    if self.capturing_hotkey {
                        if ui.button("捕获中...").clicked() {
                            self.capturing_hotkey = false;
                        }
                    } else if ui
                        .button(format!("设置: {}", self.control_hotkey))
                        .clicked()
                    {
                        self.capturing_hotkey = true;
                        self.input_listener.start_capture();
                    }
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("X目标偏移:");
                    ui.add(
                        egui::Slider::new(&mut self.control_x_target_offset, -1.0..=1.0)
                            .step_by(0.01),
                    );
                    ui.label(format!("{:.2}", self.control_x_target_offset));
                });

                ui.horizontal(|ui| {
                    ui.label("Y目标偏移:");
                    ui.add(
                        egui::Slider::new(&mut self.control_y_target_offset, -1.0..=1.0)
                            .step_by(0.01),
                    );
                    ui.label(format!("{:.2}", self.control_y_target_offset));
                });

                ui.add_space(15.0);

                if ui.button("保存控制设置").clicked()
                    && let Some(ref manager) = *self.config_manager.lock()
                {
                    let trigger_type = if self.control_trigger_toggle {
                        crate::config::TriggerType::Toggle
                    } else {
                        crate::config::TriggerType::Hold
                    };
                    let captured = self.input_listener.get_captured_key();
                    let hotkey = if let Some(key) = captured {
                        self.input_listener.stop_capture();
                        key
                    } else {
                        self.control_hotkey.clone()
                    };
                    self.control_hotkey = hotkey.clone();
                    manager.update_control(crate::config::ControlSettings {
                        yaw_sensitivity: self.control_yaw_sensitivity,
                        pitch_sensitivity: self.control_pitch_sensitivity,
                        hotkey,
                        trigger_type,
                        x_target_offset: self.control_x_target_offset,
                        y_target_offset: self.control_y_target_offset,
                    });
                    push_log(&self.logs, LogLevel::Info, "控制设置已保存");
                }
            } else if self.selected_tab == 1 {
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
                    let mut conf = *self.inferencer.conf_threshold.lock();
                    if ui.add(egui::Slider::new(&mut conf, 0.0..=1.0)).changed() {
                        *self.inferencer.conf_threshold.lock() = conf;
                    }
                });
            } else if self.selected_tab == 2 {
                ui.horizontal(|ui| {
                    ui.label("区域设置:");
                    ui.radio_value(
                        &mut self.gui_region_mode,
                        GuiRegionMode::Center640,
                        "中心 640*640",
                    );
                    ui.radio_value(
                        &mut self.gui_region_mode,
                        GuiRegionMode::Center1280,
                        "中心 1280*1280",
                    );
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

                let mut config = self.capture_config.lock();
                config.mode = self.gui_region_mode;
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
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
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
