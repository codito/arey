use once_cell::sync::Lazy;
use std::path::PathBuf;

static DEFAULT_DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
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

pub fn make_dir(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(&path)?;
    }
    Ok(())
}

pub fn get_asset_dir(suffix: &str) -> PathBuf {
    let mut path = DEFAULT_DATA_DIR.clone();
    if !suffix.is_empty() {
        path.push(suffix);
    }
    path
}

pub fn get_asset_path(asset_name: &str) -> PathBuf {
    let mut path = get_asset_dir("");
    path.push(asset_name);
    path
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_paths() {
        let asset_dir = get_asset_dir("");
        assert!(asset_dir.to_str().unwrap().contains("arey"));

        let asset_path = get_asset_path("test");
        assert!(asset_path.to_str().unwrap().contains("test"));
    }
}
