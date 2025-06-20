use console::{Style, StyledObject};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    // Prompt,
    // User,
    // AI,
    Footer,
    Error,
}

pub fn style_text(text: &str, style: MessageType) -> StyledObject<&str> {
    let style_obj = match style {
        // MessageType::Prompt => Style::new().blue().bold(),
        // MessageType::User => Style::new().blue(),
        // MessageType::AI => Style::new().white().bright(),
        MessageType::Footer => Style::new().dim(),
        MessageType::Error => Style::new().red().bold(),
    };
    style_obj.apply_to(text)
}

#[derive(Debug)]
pub struct GenerationSpinner {
    spinner: ProgressBar,
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

        Self { spinner }
    }

    pub fn clear(&self) {
        self.spinner.finish_and_clear();
    }
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
