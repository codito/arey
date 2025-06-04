use std::path::PathBuf;
use once_cell::sync::Lazy;

static DEFAULT_DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    dirs::data_local_dir()
        .map(|p| p.join("arey"))
        .unwrap_or_else(|| PathBuf::from("~/.local/share/arey"))
});

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
    DEFAULT_CONFIG_DIR.clone()
}

pub fn get_default_config() -> String {
    include_str!("../../config/default.yml").to_string()
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
