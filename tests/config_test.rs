use arey::core::config::{AreyConfigError, create_or_get_config_file, get_config};
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

// Dummy config content for tests
const DUMMY_CONFIG_CONTENT: &str = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
    n_ctx: 4096
    path: /path/to/dummy_model.gguf
  dummy-13b:
    name: dummy-13b
    type: gguf
    n_ctx: 8192
    path: /path/to/another_dummy.gguf
profiles:
  default:
    temperature: 0.7
    repeat_penalty: 1.176
    top_k: 40
    top_p: 0.1
  creative:
    temperature: 0.9
    repeat_penalty: 1.1
    top_k: 50
    top_p: 0.9
  concise:
    temperature: 0.5
    repeat_penalty: 1.2
    top_k: 30
    top_p: 0.05
chat:
  model: dummy-7b
  profile: default
task:
  model: dummy-13b
  profile: concise
"#;

// A guard struct to manage the temporary config environment.
// When this struct is dropped, it will clean up the environment variable.
struct TempConfigGuard {
    _original_xdg_config_home: Option<String>,
}

impl Drop for TempConfigGuard {
    fn drop(&mut self) {
        // SAFETY: Modifying environment variables can affect other threads or tests.
        // In this test context, we are carefully restoring the original state or
        // removing the variable if it wasn't present, ensuring isolation for tests.
        unsafe {
            if let Some(original_value) = &self._original_xdg_config_home {
                std::env::set_var("XDG_CONFIG_HOME", original_value);
            } else {
                std::env::remove_var("XDG_CONFIG_HOME");
            }
        }
    }
}

// Helper to set up a temporary config directory and file, returning a guard.
fn setup_temp_config_env(content: Option<&str>) -> (TempConfigGuard, PathBuf) {
    let temp_dir = tempdir().unwrap();
    let config_dir = temp_dir.path().join(".config").join("arey");
    let config_file = config_dir.join("arey.yml");

    // Save the current XDG_CONFIG_HOME value, if it exists
    let original_xdg_config_home = std::env::var("XDG_CONFIG_HOME").ok();

    // SAFETY: Modifying environment variables can affect other threads or tests.
    // In this test context, we are setting a temporary value that will be
    // cleaned up by `TempConfigGuard`'s `drop` implementation.
    unsafe {
        // Set XDG_CONFIG_HOME to our temporary directory to control get_config_dir
        std::env::set_var("XDG_CONFIG_HOME", temp_dir.path());
    }

    if let Some(c) = content {
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(&config_file, c).unwrap();
    }

    (
        TempConfigGuard {
            _original_xdg_config_home: original_xdg_config_home,
        },
        config_file,
    )
}

#[test]
fn test_create_or_get_config_file_when_exists() {
    let (_guard, config_file) = setup_temp_config_env(Some(DUMMY_CONFIG_CONTENT));
    let config_dir = config_file.parent().unwrap().to_path_buf();

    let (exists, file_path) = create_or_get_config_file().unwrap();

    assert!(exists);
    assert_eq!(file_path, config_file);
    assert!(config_dir.exists());
    assert!(config_file.exists());
    // _guard goes out of scope here, calling Drop and cleaning up
}

#[test]
fn test_create_or_get_config_file_when_not_exist() {
    let (_guard, config_file) = setup_temp_config_env(None); // Pass None for initial state
    let config_dir = config_file.parent().unwrap().to_path_buf();

    // Ensure the directory and file do not exist initially (already handled by setup_temp_config_env(None))
    // assert!(!config_dir.exists());
    // assert!(!config_file.exists());

    let (exists, file_path) = create_or_get_config_file().unwrap();

    assert!(!exists);
    assert_eq!(file_path, config_file);
    assert!(config_dir.exists()); // Directory should be created
    assert!(config_file.exists()); // File should be created
    // _guard goes out of scope here, calling Drop and cleaning up
}

#[test]
fn test_get_config_return_config_for_valid_schema() {
    let (_guard, _config_file) = setup_temp_config_env(Some(DUMMY_CONFIG_CONTENT));

    let config = get_config().unwrap();

    assert_eq!(config.models.len(), 2);
    assert_eq!(config.profiles.len(), 3);
    assert_eq!(config.chat.model.name, "dummy-7b");
    assert_eq!(config.chat.profile.temperature, 0.7);
    assert_eq!(config.task.model.name, "dummy-13b");
    assert_eq!(config.task.profile.temperature, 0.5);
    // _guard goes out of scope here, calling Drop and cleaning up
}

#[test]
fn test_get_config_throws_for_invalid_yaml() {
    let (_guard, _config_file) = setup_temp_config_env(Some("invalid yaml content: - ["));

    let err = get_config().unwrap_err();
    assert!(matches!(err, AreyConfigError::YAMLError(_)));
    assert!(format!("{}", err).contains("YAML parsing error"));
    // _guard goes out of scope here, calling Drop and cleaning up
}

#[test]
fn test_get_config_throws_for_missing_referenced_model() {
    let invalid_config_content = r#"
models: {} # Empty models map
profiles: {}
chat:
  model: non-existent-model # References a model not in the map
task:
  model: non-existent-model
"#;
    let (_guard, _config_file) = setup_temp_config_env(Some(invalid_config_content));

    let err = get_config().unwrap_err();
    assert!(
        matches!(err, AreyConfigError::Config(msg) if msg.contains("Model 'non-existent-model' not found"))
    );
    // _guard goes out of scope here, calling Drop and cleaning up
}

#[test]
fn test_get_config_throws_for_missing_referenced_profile() {
    let invalid_config_content = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
    n_ctx: 4096
    path: /path/to/dummy_model.gguf
profiles: {} # Empty profiles map
chat:
  model: dummy-7b
  profile: non-existent-profile # References a profile not in the map
task:
  model: dummy-7b
  profile: default # This one is fine, will use default if not found in map
"#;
    let (_guard, _config_file) = setup_temp_config_env(Some(invalid_config_content));

    let err = get_config().unwrap_err();
    assert!(
        matches!(err, AreyConfigError::Config(msg) if msg.contains("Profile 'non-existent-profile' not found"))
    );
    // _guard goes out of scope here, calling Drop and cleaning up
}
