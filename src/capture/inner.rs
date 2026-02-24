use anyhow::{Context, Result};
use windows_capture::dxgi_duplication_api::{DxgiDuplicationApi, DxgiDuplicationFormat, Error as DxgiError};
use windows_capture::monitor::Monitor;

pub struct CaptureContext {
    pub dup: DxgiDuplicationApi,
    pub width: u32,
    pub height: u32,
    pub timeout_ms: u32,
    pub rgba_buffer: Vec<u8>,
    pub rgb_buffer: Vec<u8>,
    pub nopad_buffer: Vec<u8>,
}

fn create_dup() -> Result<DxgiDuplicationApi> {
    let monitor = Monitor::primary().context("获取主显示器失败")?;
    DxgiDuplicationApi::new_options(monitor, &[DxgiDuplicationFormat::Bgra8])
        .context("创建 DXGI 复制会话失败")
}

pub fn create_capture_context(timeout_ms: u32) -> Result<CaptureContext> {
    let dup = create_dup()?;
    let width = dup.width();
    let height = dup.height();
    Ok(CaptureContext {
        dup,
        width,
        height,
        timeout_ms,
        rgba_buffer: Vec::new(),
        rgb_buffer: Vec::new(),
        nopad_buffer: Vec::new(),
    })
}

pub fn capture_full(ctx: &mut CaptureContext) -> Result<bool> {
    match ctx.dup.acquire_next_frame(ctx.timeout_ms) {
        Ok(mut frame) => {
            let buffer = frame.buffer().context("获取帧缓冲失败")?;
            let bytes = buffer.as_nopadding_buffer(&mut ctx.nopad_buffer);
            ctx.rgba_buffer.resize(bytes.len(), 0);
            ctx.rgba_buffer.copy_from_slice(bytes);
            Ok(!ctx.rgba_buffer.is_empty())
        }
        Err(DxgiError::Timeout) => Ok(false),
        Err(DxgiError::AccessLost) => {
            ctx.dup = create_dup()?;
            ctx.width = ctx.dup.width();
            ctx.height = ctx.dup.height();
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

pub fn capture_region(
    ctx: &mut CaptureContext,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<bool> {
    if width == 0 || height == 0 {
        return Ok(false);
    }
    let end_x = (x + width).min(ctx.width);
    let end_y = (y + height).min(ctx.height);
    if end_x <= x || end_y <= y {
        return Ok(false);
    }
    match ctx.dup.acquire_next_frame(ctx.timeout_ms) {
        Ok(mut frame) => {
            let buffer = frame
                .buffer_crop(x, y, end_x, end_y)
                .context("获取裁剪帧失败")?;
            let bytes = buffer.as_nopadding_buffer(&mut ctx.nopad_buffer);
            ctx.rgba_buffer.resize(bytes.len(), 0);
            ctx.rgba_buffer.copy_from_slice(bytes);
            Ok(!ctx.rgba_buffer.is_empty())
        }
        Err(DxgiError::Timeout) => Ok(false),
        Err(DxgiError::AccessLost) => {
            ctx.dup = create_dup()?;
            ctx.width = ctx.dup.width();
            ctx.height = ctx.dup.height();
            Ok(false)
        }
        Err(DxgiError::InvalidSize) => Ok(false),
        Err(e) => Err(e.into()),
    }
    }
