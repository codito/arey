//! Logging for arey.
use anyhow::Context;
use arey_core::get_data_dir;

/// Initializes the application's logging system.
///
/// This function sets up file-based logging for the application. It performs the following:
/// 1. Retrieves the application's data directory using `get_data_dir()`
/// 2. Creates a log file at `<data_dir>/arey.log`
/// 3. Implements log rotation - if the log file exceeds 100KB, it renames the existing log to `arey.log.old`
/// 4. Configures the tracing subscriber to write logs to the file
/// 5. Sets log levels: "arey" crate at DEBUG level, "rustyline" crate at INFO level
///
/// # Errors
///
/// Returns `anyhow::Result` which can contain errors from:
/// - Getting the data directory
/// - Filesystem operations (checking file metadata, renaming files)
/// - Opening/creating the log file
/// - Initializing the tracing subscriber
pub fn setup_logging() -> anyhow::Result<()> {
    let data_dir = get_data_dir().context("Failed to get data directory")?;
    let log_path = data_dir.join("arey.log");

    if log_path.exists() {
        let metadata = std::fs::metadata(&log_path)?;
        if metadata.len() > 100 * 1024 {
            // 100KB
            let backup_path = data_dir.join("arey.log.old");
            if backup_path.exists() {
                std::fs::remove_file(&backup_path)?;
            }
            std::fs::rename(&log_path, backup_path)?;
        }
    }

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    tracing_subscriber::fmt()
        .with_env_filter("arey=debug,rustyline=info")
        .with_writer(log_file)
        .init();
    Ok(())
}
