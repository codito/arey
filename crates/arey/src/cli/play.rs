use crate::{
    cli::ux::{
        ChatMessageType, TerminalRenderer, format_footer_metrics, get_theme, style_chat_text,
    },
    svc::play::{PlayFile, PlayResult},
};
use anyhow::{Context, Result};
use arey_core::{
    completion::{Completion, CompletionMetrics},
    config::Config,
};
use chrono::Local;
use futures::StreamExt;
use std::{
    fs,
    io::{Write, stdout},
    path::Path,
};

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
        "Welcome to arey play!\nEdit the play file below in your favorite editor and I'll generate a response for you. Use `Ctrl+C` to abort play session.",
    );
    println!();

    if no_watch {
        run_once(&mut play_file).await?;
        return Ok(());
    }

    // Watch the file for changes and rerun in loop
    let file_path = play_file.file_path.clone();

    println!("Watching `{}`", file_path.display());

    let (mut _watcher, mut rx) = watch::watch_file(&file_path).await?;
    let mut last_modified = fs::metadata(&file_path)?.modified()?;

    loop {
        tokio::select! {
            Some(_event) = rx.recv() => {
                // Drain subsequent events to debounce filesystem notifications.
                while rx.try_recv().is_ok() {}

                let current_modified = match fs::metadata(&file_path).and_then(|m| m.modified()) {
                    Ok(time) => time,
                    Err(_) => continue, // Couldn't get metadata, skip.
                };

                if current_modified == last_modified {
                    continue; // No change in modification time.
                }

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
                        last_modified = current_modified;
                    }
                    Err(e) => {
                        println!("{}", style_chat_text(&format!("Error reloading file: {e}"), ChatMessageType::Error));
                    }
                }
                println!();
                println!("Watching `{}`", file_path.display());
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
        let is_plain_format = matches!(
            play_file.output_settings.get("format").map(|s| s.as_str()),
            Some("plain")
        );
        let mut stream = play_file.generate().await?;

        let mut text = String::new();
        let mut finish_reason = None;
        let mut metrics = CompletionMetrics::default();

        if !is_plain_format {
            let theme = get_theme("ansi"); // TODO: Theme from config
            let mut stdout = stdout();
            let mut renderer = TerminalRenderer::new(&mut stdout, &theme);
            while let Some(chunk) = stream.next().await {
                match chunk? {
                    Completion::Response(response) => {
                        text.push_str(&response.text);
                        finish_reason = response.finish_reason;
                        renderer.render_markdown(&response.text)?;
                    }
                    Completion::Metrics(usage) => metrics = usage,
                }
            }
        } else {
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
        }

        PlayResult {
            response: text,
            metrics,
            finish_reason,
        }
    };

    play_file.result = Some(result);

    if let Some(result) = &play_file.result {
        let footer = format_footer_metrics(&result.metrics, result.finish_reason.as_deref(), false);
        println!();
        println!();
        println!("{}", style_chat_text(&footer, ChatMessageType::Footer));
    }

    Ok(())
}

pub mod watch {
    use super::*;
    use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
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
                    if matches!(event.kind, EventKind::Modify(_))
                        && tx.blocking_send(Ok(event)).is_err()
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
    use super::watch;
    use anyhow::Result;
    use notify::EventKind;
    use std::{fs, io::Write, time::Duration};
    use tempfile::tempdir;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_watch_file_sends_notification_on_modification() -> Result<()> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test_play_file.txt");
        let mut file = fs::File::create(&file_path)?;
        writeln!(file, "hello")?;
        file.sync_all()?;

        let (_watcher, mut rx) = watch::watch_file(&file_path).await?;

        // Sleep to ensure watcher is initialized
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Modify the file content to trigger a notification.
        fs::write(&file_path, "world")?;

        let event_result = timeout(Duration::from_secs(2), rx.recv()).await;

        assert!(
            event_result.is_ok(),
            "Did not receive file change notification within 2 seconds."
        );

        let event = event_result.unwrap().unwrap().unwrap();
        assert!(
            matches!(event.kind, EventKind::Modify(_)),
            "Expected a modify event, but got {event:?}",
        );

        Ok(())
    }
}
