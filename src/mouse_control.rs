#[cfg(windows)]
use windows::Win32::Foundation::POINT;
#[cfg(windows)]
use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, SetCursorPos};

#[cfg(windows)]
pub fn move_relative(dx: i32, dy: i32) {
    unsafe {
        let mut point = POINT::default();
        if GetCursorPos(&mut point).is_ok() {
            let new_x = point.x + dx;
            let new_y = point.y + dy;
            let _ = SetCursorPos(new_x, new_y);
        }
    }
}

#[cfg(not(windows))]
pub fn move_relative(_dx: i32, _dy: i32) {}

#[cfg(windows)]
#[allow(dead_code)]
pub fn get_cursor_position() -> Option<(i32, i32)> {
    unsafe {
        let mut point = POINT::default();
        if GetCursorPos(&mut point).is_ok() {
            return Some((point.x, point.y));
        }
    }
    None
}

#[cfg(not(windows))]
pub fn get_cursor_position() -> Option<(i32, i32)> {
    None
}
