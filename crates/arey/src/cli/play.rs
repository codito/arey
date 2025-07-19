use crate::{
    cli::ux::{ChatMessageType, format_footer_metrics, style_chat_text},
    svc::play::{PlayFile, PlayResult},
};
use anyhow::{Context, Result};
use arey_core::{
    completion::{Completion, CompletionMetrics},
    config::Config,
};
use chrono::Local;
use futures::StreamExt;
use std::{io::Write, path::Path};

/// Executes the play command.
///
/// This function handles the logic for the `play` subcommand, which can either
/// run once to generate a response from a play file, or watch the file for
/// changes and regenerate the response on each modification.
pub async fn execute(file: Option<&str>, no_watch: bool, config: &Config) -> Result<()> {
    let file_path =
        PlayFile::create_missing(file.map(Path::new)).context("Failed to create play file")?;
    let mut play_file = PlayFile::new(&file_path, config)?;

    println!(
        "{}",
        style_chat_text(
            "Welcome to arey play! Edit the play file below in your favorite editor and I'll generate a response for you. Use `Ctrl+C` to abort play session.",
            ChatMessageType::Footer
        )
    );
    println!();

    if no_watch {
        run_once(&mut play_file).await?;
        return Ok(());
    }

    // Watch the file for changes and rerun in loop
    let file_path = play_file.file_path.clone();

    println!(
        "{} `{}`",
        style_chat_text("Watching", ChatMessageType::Footer),
        file_path.display()
    );

    let (mut _watcher, mut rx) = watch::watch_file(&file_path).await?;

    loop {
        tokio::select! {
            Some(_event) = rx.recv() => {
                println!();
                println!(
                    "{}",
                    style_chat_text(&format!("[{}] File modified, re-generating...", Local::now().format("%Y-%m-%d %H:%M%S")), ChatMessageType::Footer)
                );
                println!();

                // Reload file content
                match PlayFile::new(&file_path, config) {
                    Ok(mut new_play_file) => {
                        // Reuse existing model if configuration hasn't changed
                        if play_file.model_config == new_play_file.model_config &&
                            play_file.model_settings == new_play_file.model_settings {

                            new_play_file.session = play_file.session.take();
                        }
                        play_file = new_play_file;

                        run_once(&mut play_file).await?;
                    }
                    Err(e) => {
                        println!("{}", style_chat_text(&format!("Error reloading file: {e}"), ChatMessageType::Error));
                    }
                }
                println!();
                println!(
                    "{} `{}`",
                    style_chat_text("Watching", ChatMessageType::Footer),
                    file_path.display()
                );
            }
            _ = tokio::signal::ctrl_c() => {
                break;
            }
        }
    }

    Ok(())
}

async fn run_once(play_file: &mut PlayFile) -> Result<()> {
    if play_file.session.is_none() {
        play_file.ensure_session().await?;
        if let Some(session_arc) = &play_file.session {
            let session = session_arc.lock().await;
            if let Some(metrics) = session.metrics() {
                println!(
                    "{} {}",
                    style_chat_text("✓ Model loaded.", ChatMessageType::Footer),
                    style_chat_text(
                        &format!("{:.2}s", metrics.init_latency_ms / 1000.0),
                        ChatMessageType::Footer
                    )
                );
                println!();
            }
        }
    }

    let result = {
        let mut stream = play_file.generate().await?;

        let mut text = String::new();
        let mut finish_reason = None;
        let mut metrics = CompletionMetrics::default();

        while let Some(chunk) = stream.next().await {
            match chunk? {
                Completion::Response(response) => {
                    text.push_str(&response.text);
                    finish_reason = response.finish_reason;

                    print!("{}", response.text);
                    std::io::stdout().flush()?;
                }
                Completion::Metrics(usage) => metrics = usage,
            }
        }

        PlayResult {
            response: text,
            metrics,
            finish_reason,
        }
    };

    play_file.result = Some(result);

    if let Some(result) = &play_file.result {
        let _ = match play_file.output_settings.get("format").map(|s| s.as_str()) {
            Some("plain") => &result.response,
            _ => &result.response,
        };

        let footer = format_footer_metrics(&result.metrics, result.finish_reason.as_deref(), false);
        println!();
        println!();
        println!("{}", style_chat_text(&footer, ChatMessageType::Footer));
    }

    Ok(())
}

pub mod watch {
    use super::*;
    use notify::{
        Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
        event::{MetadataKind, ModifyKind},
    };
    use tokio::sync::mpsc;

    /// Watches a file for changes and sends notifications on a channel.
    pub async fn watch_file(
        path: &Path,
    ) -> Result<(RecommendedWatcher, mpsc::Receiver<Result<notify::Event>>)> {
        let (tx, rx) = mpsc::channel::<Result<Event>>(128);
        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                let tx = tx.clone();
                if let Ok(event) = res {
                    if matches!(
                        event.kind,
                        EventKind::Modify(ModifyKind::Metadata(MetadataKind::Any))
                    ) && tx.blocking_send(Ok(event)).is_err()
                    {
                        // Receiver closed
                    }
                }
            },
            Config::default(),
        )?;

        watcher.watch(path, RecursiveMode::NonRecursive)?;
        Ok((watcher, rx))
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for `execute`. This would involve mocking
    // file system interactions and the watcher.

    // TODO: Add unit tests for `watch::watch_file`. This would require
    // creating temporary files and simulating file modifications.
}
