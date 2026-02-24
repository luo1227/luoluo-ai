pub mod inner;

/// Direct3D 11 屏幕捕获上下文
pub type CaptureContext = inner::CaptureContext;

/// 创建捕获上下文
pub fn create_capture_context(timeout_ms: u32) -> anyhow::Result<CaptureContext> {
    inner::create_capture_context(timeout_ms)
}

/// 捕获全屏
///
/// # 出参
/// - `bool`: 是否捕获到新帧
pub fn capture_full(ctx: &mut CaptureContext) -> anyhow::Result<bool> {
    inner::capture_full(ctx)
}

/// 捕获屏幕区域
///
/// # 入参
/// - `x`, `y`: 区域左上角坐标
/// - `width`, `height`: 区域宽高
///
/// # 出参
/// - `bool`: 是否捕获到新帧
pub fn capture_region(
    ctx: &mut CaptureContext,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> anyhow::Result<bool> {
    inner::capture_region(ctx, x, y, width, height)
}
