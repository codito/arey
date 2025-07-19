use arey_core::completion::CompletionMetrics;
use console::{Style, StyledObject};

/// Represents the type of a chat message, used for styling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatMessageType {
    /// The prompt for user input.
    Prompt,
    // User,
    // AI,
    /// Footer information, like metrics or status.
    Footer,
    /// An error message.
    Error,
}

/// Styles a string of text according to the specified `ChatMessageType`.
pub fn style_chat_text(text: &str, style: ChatMessageType) -> StyledObject<&str> {
    let style_obj = match style {
        ChatMessageType::Prompt => Style::new().blue().bold(),
        // ChatMessageType::User => Style::new().blue(),
        // ChatMessageType::AI => Style::new().white().bright(),
        ChatMessageType::Footer => Style::new().white().dim(),
        ChatMessageType::Error => Style::new().red().bold(),
    };
    style_obj.apply_to(text)
}

/// Formats the completion metrics into a string for display in the footer.
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
        details.push(format!("{tokens_per_sec:.2} tokens/s"));
    }

    // Token counts
    if metrics.completion_tokens > 0 {
        details.push(format!("{} completion tokens", metrics.completion_tokens));
    }
    if metrics.prompt_tokens > 0 {
        details.push(format!("{} prompt tokens", metrics.prompt_tokens));
    }

    let footer = if details.is_empty() {
        footer_complete
    } else {
        format!("{} {}", footer_complete, details.join(". "))
    };

    style_chat_text(&footer, ChatMessageType::Footer).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_styles() {
        let styled = style_chat_text("test", ChatMessageType::Error);
        assert_eq!(
            styled.force_styling(true).to_string(),
            "\u{1b}[31m\u{1b}[1mtest\u{1b}[0m"
        );
    }

    #[test]
    fn test_format_footer_metrics() {
        let metrics = CompletionMetrics {
            prompt_tokens: 10,
            completion_tokens: 20,
            prompt_eval_latency_ms: 100.0,
            completion_latency_ms: 200.0,
            ..Default::default()
        };

        let footer = format_footer_metrics(&metrics, Some("stop"), false);
        // The result is a styled string, so we can't do a simple assert_eq.
        assert!(footer.contains("◼ Completed (stop)."));
        assert!(footer.contains("0.10s to first token"));
        assert!(footer.contains("0.30s total"));
        assert!(footer.contains("100.00 tokens/s"));
        assert!(footer.contains("20 completion tokens"));
        assert!(footer.contains("10 prompt tokens"));

        let cancelled_footer = format_footer_metrics(&metrics, None, true);
        assert_eq!("◼ Cancelled.", cancelled_footer);
    }
}
