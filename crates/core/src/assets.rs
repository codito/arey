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

pub fn get_data_dir() -> std::io::Result<PathBuf> {
    let path = if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg_data_home).join("arey")
    } else {
        _DEFAULT_DATA_DIR.clone()
    };
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn get_default_config() -> String {
    include_str!("../data/config.yml").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;

    // Mutex to serialize tests that modify the environment
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_get_config_dir_with_xdg_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let tmp_dir = tempfile::tempdir().unwrap();
        let xdg_config_path = tmp_dir.path();
        unsafe {
            env::set_var("XDG_CONFIG_HOME", xdg_config_path);
        }

        let config_dir = get_config_dir();
        assert_eq!(config_dir, xdg_config_path.join("arey"));

        unsafe {
            env::remove_var("XDG_CONFIG_HOME");
        }
    }

    #[test]
    fn test_get_config_dir_without_xdg_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::remove_var("XDG_CONFIG_HOME");
        }
        let config_dir = get_config_dir();
        let expected = dirs::config_dir()
            .map(|p| p.join("arey"))
            .unwrap_or_else(|| PathBuf::from("~/.config/arey"));
        assert_eq!(config_dir, expected);
    }

    #[test]
    fn test_get_default_config() {
        let config = get_default_config();
        assert!(!config.is_empty());
        assert!(config.contains("models:"));
        assert!(config.contains("profiles:"));
    }

    #[test]
    fn test_get_data_dir_with_xdg_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let tmp_dir = tempfile::tempdir().unwrap();
        let xdg_data_path = tmp_dir.path();
        unsafe {
            env::set_var("XDG_DATA_HOME", xdg_data_path);
        }

        let data_dir = get_data_dir().unwrap();
        assert_eq!(data_dir, xdg_data_path.join("arey"));

        unsafe {
            env::remove_var("XDG_DATA_HOME");
        }
    }

    #[test]
    fn test_get_data_dir_without_xdg_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::remove_var("XDG_DATA_HOME");
        }
        let data_dir = get_data_dir().unwrap();
        let expected = dirs::data_local_dir()
            .map(|p| p.join("arey"))
            .unwrap_or_else(|| PathBuf::from("~/.local/share/arey"));
        assert_eq!(data_dir, expected);
    }
}
