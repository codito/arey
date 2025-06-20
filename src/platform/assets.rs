use once_cell::sync::Lazy;
use std::path::PathBuf;

static _DEFAULT_DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    dirs::data_local_dir()
        .map(|p| p.join("arey"))
        .unwrap_or_else(|| PathBuf::from("~/.local/share/arey"))
});

// DEFAULT_CONFIG_DIR is now a fallback, as get_config_dir will check XDG_CONFIG_HOME first
static DEFAULT_CONFIG_DIR: Lazy<PathBuf> = Lazy::new(|| {
    dirs::config_dir()
        .map(|p| p.join("arey"))
        .unwrap_or_else(|| PathBuf::from("~/.config/arey"))
});

pub fn get_config_dir() -> PathBuf {
    // Check XDG_CONFIG_HOME first, then fall back to default
    if let Ok(xdg_config_home) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg_config_home).join("arey")
    } else {
        DEFAULT_CONFIG_DIR.clone()
    }
}

pub fn get_default_config() -> String {
    include_str!("../data/config.yml").to_string()
}

pub fn get_default_play_file() -> String {
    include_str!("../data/play.md").to_string()
}
