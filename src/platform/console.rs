use console::{Style, StyledObject};
use indicatif::{ProgressBar, ProgressStyle};
use crate::core::completion::CompletionMetrics;

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

pub fn format_footer_metrics(
    metrics: &CompletionMetrics,
    finish_reason: Option<&str>,
    is_cancelled: bool,
) -> String {
    if is_cancelled {
        return "◼ Cancelled.".to_string();
    }

    let mut footer_complete = String::from("◼ Completed");
    if let Some(reason) = finish_reason {
        footer_complete.push_str(&format!(" ({reason})"));
    }
    footer_complete.push('.');

    let mut details = Vec::new();

    // Time metrics
    if metrics.prompt_eval_latency_ms > 0.0 {
        details.push(format!(
            "{:.2}s to first token",
            metrics.prompt_eval_latency_ms / 1000.0
        ));
    }
    if metrics.completion_latency_ms > 0.0 {
        details.push(format!(
            "{:.2}s total",
            (metrics.prompt_eval_latency_ms + metrics.completion_latency_ms) / 1000.0
        ));
    }

    // Tokens/s rate
    if metrics.completion_tokens > 0 && metrics.completion_latency_ms > 0.0 {
        let tokens_per_sec =
            metrics.completion_tokens as f32 * 1000.0 / metrics.completion_latency_ms;
        details.push(format!("{:.2} tokens/s", tokens_per_sec));
    }

    // Token counts
    if metrics.completion_tokens > 0 {
        details.push(format!("{} completion tokens", metrics.completion_tokens));
    }
    if metrics.prompt_tokens > 0 {
        details.push(format!("{} prompt tokens", metrics.prompt_tokens));
    }

    if details.is_empty() {
        footer_complete
    } else {
        format!("{} {}", footer_complete, details.join(". "))
    }
}
