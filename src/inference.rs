use std::sync::Arc;
use anyhow::Context;
use parking_lot::Mutex;
use tracing::error;
use usls::{Config, Device, DType, DynConf, Image, Model, Y, YOLO};
use windows_capture::{
    capture::{Context as CaptureContext, GraphicsCaptureApiHandler},
    frame::Frame,
    graphics_capture_api::InternalCaptureControl,
    monitor::Monitor,
    settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    },
};

pub struct InferenceEngine {
    model: Arc<Mutex<Option<(YOLO, usls::Engines)>>>,
    pub results: Arc<Mutex<Vec<Y>>>,
    pub is_running: Arc<Mutex<bool>>,
    pub conf_threshold: Arc<Mutex<f32>>,
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            results: Arc::new(Mutex::new(Vec::new())),
            is_running: Arc::new(Mutex::new(false)),
            conf_threshold: Arc::new(Mutex::new(0.25)),
        }
    }

    pub fn load_model(&self, path: std::path::PathBuf, device_type: &str) -> anyhow::Result<()> {
        let device = match device_type {
            "cpu" => Device::Cpu(0),
            "cuda" => Device::Cuda(0),
            "tensorrt" => Device::TensorRt(0),
            "nvrtx" => Device::NvRtx(0),
            _ => Device::Cpu(0),
        };

        let config = Config::yolo()
            .with_model_file(path.to_string_lossy())
            .with_model_device(device)
            .with_model_dtype(DType::Fp16)
            .commit()
            .with_context(|| format!("USLS 配置提交失败: path={:?}, device_type={}", path, device_type))?;

        let (model, engines) = YOLO::build(config)
            .with_context(|| format!("YOLO 模型构建失败: path={:?}, device_type={}", path, device_type))?;
        *self.model.lock() = Some((model, engines));
        Ok(())
    }
}

struct CaptureProcessor {
    model: Arc<Mutex<Option<(YOLO, usls::Engines)>>>,
    results: Arc<Mutex<Vec<Y>>>,
    conf_threshold: Arc<Mutex<f32>>,
}

impl GraphicsCaptureApiHandler for CaptureProcessor {
    type Flags = (
        Arc<Mutex<Option<(YOLO, usls::Engines)>>>,
        Arc<Mutex<Vec<Y>>>,
        Arc<Mutex<f32>>,
    );

    type Error = anyhow::Error;

    fn new(ctx: CaptureContext<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            model: ctx.flags.0,
            results: ctx.flags.1,
            conf_threshold: ctx.flags.2,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let mut model_lock = self.model.lock();
        if let Some((model, engines)) = model_lock.as_mut() {
            let width = frame.width();
            let height = frame.height();
            let mut buffer = frame.buffer()?;
            let data = buffer.as_nopadding_buffer()?;

            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in data.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }

            let img = Image::from_u8s(&rgb, width, height)?;

            let conf = *self.conf_threshold.lock();
            model.confs = DynConf::new_or(&[conf], model.nc, conf);

            let ys = model.run(engines, &[img])?;
            if let Some(y) = ys.into_iter().next() {
                *self.results.lock() = vec![y];
            }
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub fn start_capture(engine: Arc<InferenceEngine>) -> anyhow::Result<()> {
    let primary_monitor = Monitor::primary()?;
    let settings = Settings::new(
        primary_monitor,
        CursorCaptureSettings::Default,
        DrawBorderSettings::Default,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        ColorFormat::Rgba8,
        (
            engine.model.clone(),
            engine.results.clone(),
            engine.conf_threshold.clone(),
        ),
    );

    std::thread::spawn(move || {
        if let Err(e) = CaptureProcessor::start(settings) {
            error!("Screen Capture Failed: {}", e);
        }
    });

    Ok(())
}
