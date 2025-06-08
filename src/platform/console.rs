use console::{Style, StyledObject, Term};
use ctrlc;
use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

static SHOULD_STOP: AtomicBool = AtomicBool::new(false);
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

pub struct SignalContext {
    stopped: Arc<AtomicBool>,
}

impl SignalContext {
    pub fn new() -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let s = stopped.clone();

        ctrlc::set_handler(move || {
            s.store(true, Ordering::Relaxed);
        })
        .expect("Error setting Ctrl-C handler");

        SignalContext { stopped }
    }

    pub fn should_stop(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }
}

pub struct Spinner(ProgressBar);

impl Spinner {
    pub fn new() -> Self {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        Spinner(spinner)
    }

    pub fn set_message(&self, msg: &str) {
        self.0.set_message(msg.to_string());
    }

    pub fn tick(&self) {
        self.0.tick();
    }

    pub fn finish(&self) {
        self.0.finish_and_clear();
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
