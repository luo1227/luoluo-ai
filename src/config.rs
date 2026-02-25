use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum TriggerType {
    #[default]
    Hold,
    Toggle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlSettings {
    pub yaw_sensitivity: f32,
    pub pitch_sensitivity: f32,
    pub hotkey: String,
    pub trigger_type: TriggerType,
    pub x_target_offset: f32,
    pub y_target_offset: f32,
}

impl Default for ControlSettings {
    fn default() -> Self {
        Self {
            yaw_sensitivity: 0.3,
            pitch_sensitivity: 0.3,
            hotkey: "x1".to_string(),
            trigger_type: TriggerType::Hold,
            x_target_offset: 0.0,
            y_target_offset: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub control: ControlSettings,
}

pub struct ConfigManager {
    config: Arc<Mutex<AppConfig>>,
    path: PathBuf,
}

impl ConfigManager {
    pub fn new(path: PathBuf) -> Self {
        let config = if path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_default())
                .unwrap_or_default()
        } else {
            AppConfig::default()
        };
        Self {
            config: Arc::new(Mutex::new(config)),
            path,
        }
    }

    pub fn load(&self) -> AppConfig {
        self.config.lock().clone()
    }

    pub fn update_control(&self, control: ControlSettings) {
        let mut config = self.config.lock();
        config.control = control;
        if let Ok(json) = serde_json::to_string_pretty(&*config) {
            let _ = std::fs::write(&self.path, json);
        }
    }
}
