use crate::play;
use anyhow::{Context, Result};
use arey_core::config::Config;
use std::path::Path;

pub async fn execute(file: Option<&str>, no_watch: bool, config: &Config) -> Result<()> {
    let file_path = play::PlayFile::create_missing(file.map(Path::new))
        .context("Failed to create play file")?;
    let mut play_file = play::PlayFile::new(&file_path, config)?;
    play::run_play(&mut play_file, config, no_watch).await
}
