use anyhow::{Context, Result};
use windows_capture::dxgi_duplication_api::{
    DxgiDuplicationApi, DxgiDuplicationFormat, Error as DxgiError,
};
use windows_capture::monitor::Monitor;

pub struct ScreenCapture {
    dup: DxgiDuplicationApi,
    width: u32,
    height: u32,
    timeout_ms: u32,
    pub rgba_buffer: Vec<u8>,
    #[allow(dead_code)]
    pub rgb_buffer: Vec<u8>,
    pub nopad_buffer: Vec<u8>,
}

impl ScreenCapture {
    pub fn try_new(timeout_ms: u32) -> Result<Self> {
        let monitor = Monitor::primary().context("获取主显示器失败")?;
        let dup = DxgiDuplicationApi::new_options(monitor, &[DxgiDuplicationFormat::Bgra8])
            .context("创建 DXGI 复制会话失败")?;
        let width = dup.width();
        let height = dup.height();
        Ok(Self {
            dup,
            width,
            height,
            timeout_ms,
            rgba_buffer: Vec::new(),
            rgb_buffer: Vec::new(),
            nopad_buffer: Vec::new(),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn recreate(&mut self) -> Result<()> {
        let monitor = Monitor::primary().context("获取主显示器失败")?;
        self.dup = DxgiDuplicationApi::new_options(monitor, &[DxgiDuplicationFormat::Bgra8])
            .context("创建 DXGI 复制会话失败")?;
        self.width = self.dup.width();
        self.height = self.dup.height();
        Ok(())
    }

    pub fn capture_full(&mut self) -> Result<bool> {
        match self.dup.acquire_next_frame(self.timeout_ms) {
            Ok(mut frame) => {
                let buffer = frame.buffer().context("获取帧缓冲失败")?;
                let bytes = buffer.as_nopadding_buffer(&mut self.nopad_buffer);
                self.rgba_buffer.resize(bytes.len(), 0);
                self.rgba_buffer.copy_from_slice(bytes);
                Ok(!self.rgba_buffer.is_empty())
            }
            Err(DxgiError::Timeout) => Ok(false),
            Err(DxgiError::AccessLost) => {
                self.recreate()?;
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn capture_region(&mut self, x: u32, y: u32, width: u32, height: u32) -> Result<bool> {
        if width == 0 || height == 0 {
            return Ok(false);
        }
        let end_x = (x + width).min(self.width);
        let end_y = (y + height).min(self.height);
        if end_x <= x || end_y <= y {
            return Ok(false);
        }
        match self.dup.acquire_next_frame(self.timeout_ms) {
            Ok(mut frame) => {
                let buffer = frame
                    .buffer_crop(x, y, end_x, end_y)
                    .context("获取裁剪帧失败")?;
                let bytes = buffer.as_nopadding_buffer(&mut self.nopad_buffer);
                self.rgba_buffer.resize(bytes.len(), 0);
                self.rgba_buffer.copy_from_slice(bytes);
                Ok(!self.rgba_buffer.is_empty())
            }
            Err(DxgiError::Timeout) => Ok(false),
            Err(DxgiError::AccessLost) => {
                self.recreate()?;
                Ok(false)
            }
            Err(DxgiError::InvalidSize) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}
