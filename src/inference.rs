use std::sync::Arc;
use std::time::Instant;
use anyhow::{bail, Context};
use parking_lot::Mutex;
use usls::{Config, Device, DType, DynConf, Image, Model, Task, Y, YOLO};
use crate::capture::CaptureContext;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegionMode {
    Fullscreen,
    Center640,
    Center1280,
    Custom { x: u32, y: u32, width: u32, height: u32 },
}

#[derive(Clone, Copy)]
pub struct CaptureConfig {
    pub mode: RegionMode,
}

pub struct InferenceEngine {
    model: Arc<Mutex<Option<(YOLO, usls::Engines)>>>,
    pub results: Arc<Mutex<Vec<Y>>>,
    pub is_running: Arc<Mutex<bool>>,
    pub conf_threshold: Arc<Mutex<f32>>,
    pub capture_config: Arc<Mutex<CaptureConfig>>,
}

pub struct CaptureStats {
    pub did_capture: bool,
    pub capture_ms: f64,
    pub infer_ms: f64,
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            results: Arc::new(Mutex::new(Vec::new())),
            is_running: Arc::new(Mutex::new(false)),
            conf_threshold: Arc::new(Mutex::new(0.25)),
            capture_config: Arc::new(Mutex::new(CaptureConfig {
                mode: RegionMode::Fullscreen,
            })),
        }
    }

    pub fn load_model(
        &self,
        path: std::path::PathBuf,
        device_type: &str,
        version_override: Option<u8>,
    ) -> anyhow::Result<()> {
        if path.as_os_str().is_empty() {
            bail!("模型路径为空");
        }
        if !path.exists() {
            bail!("模型文件不存在: {}", path.display());
        }
        if !path.is_file() {
            bail!("模型路径不是文件: {}", path.display());
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
            != Some("onnx")
        {
            bail!("模型文件不是 .onnx: {}", path.display());
        }

        let device = match device_type {
            "cpu" => Device::Cpu(0),
            "cuda" => Device::Cuda(0),
            "tensorrt" => Device::TensorRt(0),
            _ => Device::Cpu(0),
        };

        let (task, version) = infer_yolo_meta(&path, version_override)?;
        let dtype = match device {
            Device::Cpu(_) => DType::Fp32,
            _ => DType::Auto,
        };

        let config = Config::yolo()
            .with_task(task)
            .with_version(version.into())
            .with_model_file(path.to_string_lossy())
            .with_model_device(device)
            .with_model_dtype(dtype)
            .commit()
            .with_context(|| format!("USLS 配置提交失败: path={:?}, device_type={}", path, device_type))?;

        let (model, engines) = YOLO::build(config)
            .with_context(|| format!("YOLO 模型构建失败: path={:?}, device_type={}", path, device_type))?;
        *self.model.lock() = Some((model, engines));
        Ok(())
    }

    pub fn capture_once(&self, ctx: &mut CaptureContext) -> anyhow::Result<CaptureStats> {
        if !*self.is_running.lock() {
            return Ok(CaptureStats {
                did_capture: false,
                capture_ms: 0.0,
                infer_ms: 0.0,
            });
        }
        let capture_start = Instant::now();
        let config = *self.capture_config.lock();
        let full_w = ctx.width;
        let full_h = ctx.height;

        let (did_capture, width, height) = match config.mode {
            RegionMode::Fullscreen => {
                let did = crate::capture::capture_full(ctx).context("DXGI 全屏捕获失败")?;
                (did, full_w, full_h)
            }
            RegionMode::Center640 => {
                let w = 640.min(full_w);
                let h = 640.min(full_h);
                let x = (full_w - w) / 2;
                let y = (full_h - h) / 2;
                let did = crate::capture::capture_region(ctx, x, y, w, h).context("DXGI 中心区域捕获失败")?;
                (did, w, h)
            }
            RegionMode::Center1280 => {
                let w = 1280.min(full_w);
                let h = 1280.min(full_h);
                let x = (full_w - w) / 2;
                let y = (full_h - h) / 2;
                let did = crate::capture::capture_region(ctx, x, y, w, h).context("DXGI 中心区域捕获失败")?;
                (did, w, h)
            }
            RegionMode::Custom { x, y, width, height } => {
                if width > 0 && height > 0 && x < full_w && y < full_h {
                    let w = width.min(full_w - x);
                    let h = height.min(full_h - y);
                    let did = crate::capture::capture_region(ctx, x, y, w, h).context("DXGI 自定义区域捕获失败")?;
                    (did, w, h)
                } else {
                    let did = crate::capture::capture_full(ctx).context("DXGI 全屏捕获失败")?;
                    (did, full_w, full_h)
                }
            }
        };
        if !did_capture || ctx.rgba_buffer.is_empty() {
            return Ok(CaptureStats {
                did_capture: false,
                capture_ms: 0.0,
                infer_ms: 0.0,
            });
        }

        let mut capture_ms = 0.0;
        let mut infer_ms = 0.0;
        let mut model_lock = self.model.lock();
        if let Some((model, engines)) = model_lock.as_mut() {
            let rgb_len = (width * height * 3) as usize;
            ctx.rgb_buffer.resize(rgb_len, 0);
            
            // 使用 Rayon 并行加速 RGBA -> RGB 转换
            let rgba = &ctx.rgba_buffer;
            let rgb = &mut ctx.rgb_buffer;
            
            rgb.par_chunks_exact_mut(3)
                .zip(rgba.par_chunks_exact(4))
                .for_each(|(rgb_p, rgba_p)| {
                    rgb_p[0] = rgba_p[2]; // R (DXGI is BGRA)
                    rgb_p[1] = rgba_p[1]; // G
                    rgb_p[2] = rgba_p[0]; // B
                });

            let img = Image::from_u8s(&ctx.rgb_buffer, width, height)?;
            capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;
            let conf = *self.conf_threshold.lock();
            model.confs = DynConf::new_or(&[conf], model.nc, conf);

            let infer_start = Instant::now();
            let ys = model.run(engines, &[img])?;
            infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
            if let Some(y) = ys.into_iter().next() {
                *self.results.lock() = vec![y];
            }
        }

        Ok(CaptureStats {
            did_capture,
            capture_ms,
            infer_ms,
        })
    }
}

fn infer_yolo_meta(path: &std::path::PathBuf, version_override: Option<u8>) -> anyhow::Result<(Task, u8)> {
    let stem = path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let task = if stem.contains("seg") || stem.contains("segment") {
        Task::InstanceSegmentation
    } else if stem.contains("pose") || stem.contains("kpt") {
        Task::KeypointsDetection
    } else if stem.contains("obb") {
        Task::OrientedObjectDetection
    } else if stem.contains("cls") || stem.contains("classify") {
        Task::ImageClassification
    } else {
        Task::ObjectDetection
    };

    if let Some(ver) = version_override.or_else(|| parse_version_from_name(&stem)) {
        return Ok((task, ver));
    }

    bail!(
        "无法从文件名解析 YOLO 版本，请确保文件名包含 yolo8/yolo11/yolo26 或 v8/v11/v26: {}",
        path.display()
    )
}

fn parse_version_from_name(name: &str) -> Option<u8> {
    if let Some(idx) = name.find("yolo") {
        let digits: String = name[idx + 4..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(v) = digits.parse::<u8>() {
            return Some(v);
        }
    }
    let mut iter = name.chars().peekable();
    while let Some(ch) = iter.next() {
        if ch == 'v' {
            let mut digits = String::new();
            while let Some(next) = iter.peek() {
                if next.is_ascii_digit() {
                    digits.push(*next);
                    iter.next();
                } else {
                    break;
                }
            }
            if let Ok(v) = digits.parse::<u8>() {
                return Some(v);
            }
        }
    }
    None
}
