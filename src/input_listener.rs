use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use parking_lot::Mutex;

#[cfg(windows)]
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON, VK_XBUTTON1, VK_XBUTTON2,
};

pub struct InputListener {
    running: Arc<AtomicBool>,
    hotkey_pressed: Arc<Mutex<bool>>,
    hotkey_name: Arc<Mutex<String>>,
    last_trigger: Arc<Mutex<Instant>>,
    capturing: Arc<AtomicBool>,
    captured_key: Arc<Mutex<Option<String>>>,
    toggle_state: Arc<Mutex<bool>>,
}

impl InputListener {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            hotkey_pressed: Arc::new(Mutex::new(false)),
            hotkey_name: Arc::new(Mutex::new("x1".to_string())),
            last_trigger: Arc::new(Mutex::new(Instant::now())),
            capturing: Arc::new(AtomicBool::new(false)),
            captured_key: Arc::new(Mutex::new(None)),
            toggle_state: Arc::new(Mutex::new(false)),
        }
    }

    pub fn set_hotkey(&self, name: &str) {
        *self.hotkey_name.lock() = name.to_lowercase();
    }

    pub fn start(&self) {
        if self.running.load(Ordering::SeqCst) {
            return;
        }
        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();
        let hotkey_pressed = self.hotkey_pressed.clone();
        let hotkey_name = self.hotkey_name.clone();

        thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                let name = hotkey_name.lock().clone();
                let pressed = Self::check_hotkey(&name);
                *hotkey_pressed.lock() = pressed;
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_hotkey_pressed(&self) -> bool {
        *self.hotkey_pressed.lock()
    }

    #[allow(dead_code)]
    pub fn get_toggle_state(&self) -> bool {
        *self.toggle_state.lock()
    }

    #[allow(dead_code)]
    pub fn set_toggle_state(&self, state: bool) {
        *self.toggle_state.lock() = state;
    }

    pub fn check_trigger_cooldown(&self, cooldown_ms: u64) -> bool {
        let last = *self.last_trigger.lock();
        if last.elapsed().as_millis() as u64 >= cooldown_ms {
            *self.last_trigger.lock() = Instant::now();
            true
        } else {
            false
        }
    }

    pub fn start_capture(&self) {
        self.capturing.store(true, Ordering::SeqCst);
        *self.captured_key.lock() = None;

        let capturing = self.capturing.clone();
        let captured_key = self.captured_key.clone();

        thread::spawn(move || {
            while capturing.load(Ordering::SeqCst) {
                if let Some(key) = Self::capture_any_key() {
                    *captured_key.lock() = Some(key);
                    capturing.store(false, Ordering::SeqCst);
                    break;
                }
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });
    }

    pub fn stop_capture(&self) {
        self.capturing.store(false, Ordering::SeqCst);
    }

    pub fn get_captured_key(&self) -> Option<String> {
        self.captured_key.lock().clone()
    }

    #[allow(dead_code)]
    pub fn is_capturing(&self) -> bool {
        self.capturing.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    #[cfg(windows)]
    fn capture_any_key() -> Option<String> {
        unsafe {
            if GetAsyncKeyState(VK_LBUTTON.0 as i32) as u16 & 0x8000 != 0 {
                return Some("left".to_string());
            }
            if GetAsyncKeyState(VK_RBUTTON.0 as i32) as u16 & 0x8000 != 0 {
                return Some("right".to_string());
            }
            if GetAsyncKeyState(VK_MBUTTON.0 as i32) as u16 & 0x8000 != 0 {
                return Some("middle".to_string());
            }
            if GetAsyncKeyState(VK_XBUTTON1.0 as i32) as u16 & 0x8000 != 0 {
                return Some("x1".to_string());
            }
            if GetAsyncKeyState(VK_XBUTTON2.0 as i32) as u16 & 0x8000 != 0 {
                return Some("x2".to_string());
            }
            None
        }
    }

    #[cfg(not(windows))]
    fn capture_any_key() -> Option<String> {
        None
    }

    #[cfg(windows)]
    fn check_hotkey(name: &str) -> bool {
        unsafe {
            match name {
                "x1" | "mouse_x1" | "侧键1" => {
                    GetAsyncKeyState(VK_XBUTTON1.0 as i32) as u16 & 0x8000 != 0
                }
                "x2" | "mouse_x2" | "侧键2" => {
                    GetAsyncKeyState(VK_XBUTTON2.0 as i32) as u16 & 0x8000 != 0
                }
                "left" | "左键" => GetAsyncKeyState(VK_LBUTTON.0 as i32) as u16 & 0x8000 != 0,
                "right" | "右键" => GetAsyncKeyState(VK_RBUTTON.0 as i32) as u16 & 0x8000 != 0,
                "middle" | "中键" => GetAsyncKeyState(VK_MBUTTON.0 as i32) as u16 & 0x8000 != 0,
                _ => false,
            }
        }
    }

    #[cfg(not(windows))]
    fn check_hotkey(_name: &str) -> bool {
        false
    }
}

impl Default for InputListener {
    fn default() -> Self {
        Self::new()
    }
}
