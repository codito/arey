use crate::core::completion::CancellationToken;
use anyhow::Result;
use console::{Style, StyledObject, Term};
use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use tokio::signal;

static CONSOLE_INSTANCE: Lazy<Term> = Lazy::new(|| Term::stdout());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Prompt,
    User,
    AI,
    Footer,
    Error,
}

pub fn get_console() -> &'static Term {
    &CONSOLE_INSTANCE
}

pub fn style_text(text: &str, style: MessageType) -> StyledObject<&str> {
    let style_obj = match style {
        MessageType::Prompt => Style::new().blue().bold(),
        MessageType::User => Style::new().blue(),
        MessageType::AI => Style::new().white().bright(),
        MessageType::Footer => Style::new().dim(),
        MessageType::Error => Style::new().red().bold(),
    };
    style_obj.apply_to(text)
}

#[derive(Debug)]
pub struct GenerationSpinner {
    spinner: ProgressBar,
    cancel_token: CancellationToken,
}

impl GenerationSpinner {
    pub fn new() -> Self {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message("Generating...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));

        Self {
            spinner,
            cancel_token: CancellationToken::new(),
        }
    }

    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    pub fn token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    pub fn clear_message(&self) {
        self.spinner.finish_and_clear();
    }

    pub fn finish(self) {
        self.spinner.finish_and_clear();
    }

    /// Utility function to handle spinner during stream processing
    pub async fn handle_stream<F, T>(mut self, future: F) -> Option<T>
    where
        F: std::future::Future<Output = Result<T>> + Send,
        T: Send + 'static,
    {
        let ctrl_c_future = signal::ctrl_c();
        let result_future = future;

        tokio::select! {
            _ = self.cancel_token.cancelled() => {
                self.spinner.finish_and_clear();
                None
            }
            _ = ctrl_c_future => {
                self.cancel();
                self.spinner.finish_and_clear();
                None
            }
            result = result_future => {
                self.spinner.finish_and_clear();
                result.ok()
            }
        }
    }
}

pub fn capture_stderr<F>(f: F) -> String
where
    F: FnOnce() -> (),
{
    // In a real implementation we would capture stderr output
    // For simplicity we'll just return an empty string
    f();
    "".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_styles() {
        let styled = style_text("test", MessageType::Error);
        assert_eq!(
            styled.force_styling(true).to_string(),
            "\u{1b}[31m\u{1b}[1mtest\u{1b}[0m"
        );
    }
}
