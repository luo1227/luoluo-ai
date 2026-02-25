use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use usls::{Config, DType, Device, DynConf, Image, Model, Task, Y, YOLO};

pub struct YoloInferencer {
    model: Arc<Mutex<Option<(YOLO, usls::Engines)>>>,
    pub results: Arc<Mutex<Vec<Y>>>,
    pub conf_threshold: Arc<Mutex<f32>>,
}

impl YoloInferencer {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            results: Arc::new(Mutex::new(Vec::new())),
            conf_threshold: Arc::new(Mutex::new(0.25)),
        }
    }

    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        self.model.lock().is_some()
    }

    pub fn load_model(&self, path: &Path, device_type: &str, version: u8) -> Result<()> {
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

        let dtype = match device {
            Device::Cpu(_) => DType::Fp32,
            _ => DType::Auto,
        };

        let config = Config::yolo()
            .with_task(Task::ObjectDetection)
            .with_version(version.into())
            .with_model_file(path.to_string_lossy())
            .with_model_device(device)
            .with_model_dtype(dtype)
            .commit()
            .with_context(|| {
                format!(
                    "USLS 配置提交失败: path={:?}, device_type={}",
                    path, device_type
                )
            })?;

        let (model, engines) = YOLO::build(config).with_context(|| {
            format!(
                "YOLO 模型构建失败: path={:?}, device_type={}",
                path, device_type
            )
        })?;
        *self.model.lock() = Some((model, engines));
        Ok(())
    }

    pub fn infer(&self, rgb_buffer: &[u8], width: u32, height: u32) -> Result<Vec<Y>> {
        let mut model_lock = self.model.lock();
        let (model, engines) = model_lock
            .as_mut()
            .context("模型未加载，请先调用 load_model")?;

        let conf = *self.conf_threshold.lock();
        model.confs = DynConf::new_or(&[conf], model.nc, conf);

        let img = Image::from_u8s(rgb_buffer, width, height)?;
        let ys = model.run(engines, &[img])?;

        let results: Vec<Y> = ys.into_iter().collect();
        *self.results.lock() = results.clone();
        Ok(results)
    }

    pub fn infer_with_preprocess(
        &self,
        rgba_buffer: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Y>> {
        let rgb_len = (width * height * 3) as usize;
        let mut rgb_buffer = vec![0u8; rgb_len];

        rgba_buffer
            .par_chunks_exact(4)
            .zip(rgb_buffer.par_chunks_exact_mut(3))
            .for_each(|(rgba_p, rgb_p)| {
                rgb_p[0] = rgba_p[2];
                rgb_p[1] = rgba_p[1];
                rgb_p[2] = rgba_p[0];
            });

        self.infer(&rgb_buffer, width, height)
    }
}

impl Default for YoloInferencer {
    fn default() -> Self {
        Self::new()
    }
}
