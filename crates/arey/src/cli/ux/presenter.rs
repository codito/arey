use arey_core::completion::CompletionMetrics;
use console::{Style, StyledObject};

/// Represents the type of a chat message, used for styling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatMessageType {
    /// The prompt for user input.
    Prompt,
    // User,
    // AI,
    /// Prompt metadata line
    PromptMeta,
    /// Footer information, like metrics or status.
    Footer,
    /// An error message.
    Error,
}

/// Styles a string of text according to the specified `ChatMessageType`.
pub fn style_chat_text(text: &str, style: ChatMessageType) -> StyledObject<&str> {
    let style_obj = match style {
        ChatMessageType::PromptMeta => Style::new().blue(),
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

    let mut details = Vec::new();

    // Combined time and speed: "0.30s (0.10s first) | 100 tok/s"
    let total_time_ms = metrics.prompt_eval_latency_ms + metrics.completion_latency_ms;
    let first_token_ms = metrics.prompt_eval_latency_ms;
    let tokens_per_sec = if metrics.completion_tokens > 0 && metrics.completion_latency_ms > 0.0 {
        Some(metrics.completion_tokens as f32 * 1000.0 / metrics.completion_latency_ms)
    } else {
        None
    };

    if total_time_ms > 0.0 {
        let mut time_str = format!("{:.2}s", total_time_ms / 1000.0);

        // Add first token time in parentheses
        if first_token_ms > 0.0 {
            time_str.push_str(&format!(" ({:.2}s first)", first_token_ms / 1000.0));
        }

        details.push(time_str);

        // Add tokens/s rate
        if let Some(tps) = tokens_per_sec {
            details.push(format!("{:.0} tok/s", tps));
        }
    }

    // Token counts: "10 in (5 cache) | 20 out | 30/2048 ctx"
    let prompt = metrics.prompt_tokens;
    let completion = metrics.completion_tokens;
    let cached = metrics
        .cache_metrics
        .as_ref()
        .and_then(|c| c.tokens_skipped);
    let n_ctx = metrics.cache_metrics.as_ref().and_then(|c| c.n_ctx);

    if prompt > 0 || completion > 0 {
        let mut token_info = String::new();

        // "10 in (5 cache)" or just "10 in"
        if prompt > 0 {
            token_info.push_str(&format!("{} tok in", prompt));
            if let Some(cached) = cached
                && cached > 0
            {
                token_info.push_str(&format!(" ({} cache)", cached));
            }
        }

        // " | 20 out"
        if completion > 0 {
            if !token_info.is_empty() {
                token_info.push_str(" | ");
            }
            token_info.push_str(&format!("{} tok out", completion));
        }

        // " | 30/2048 ctx"
        if let Some(n_ctx) = n_ctx {
            if !token_info.is_empty() {
                token_info.push_str(" | ");
            }
            let total_tokens = prompt + completion;
            token_info.push_str(&format!("{}/{} ctx", total_tokens, n_ctx));
        }

        if !token_info.is_empty() {
            details.push(token_info);
        }
    }

    let footer = if details.is_empty() {
        footer_complete
    } else {
        format!("{} | {}", footer_complete, details.join(" | "))
    };

    style_chat_text(&footer, ChatMessageType::Footer).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::completion::CacheMetrics;

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
            cache_metrics: Some(CacheMetrics {
                cache_hit: false,
                strategy: Some("Hybrid".to_string()),
                tokens_skipped: None,
                n_ctx: Some(2048),
                checkpoint_transition: None,
            }),
            ..Default::default()
        };

        let footer = format_footer_metrics(&metrics, Some("stop"), false);
        assert!(footer.contains("◼ Completed (stop)"));
        assert!(footer.contains("0.30s (0.10s first)"));
        assert!(footer.contains("100 tok/s"));
        assert!(footer.contains("10 tok in"));
        assert!(footer.contains("20 tok out"));
        assert!(footer.contains("30/2048 ctx"));

        // Test with cache
        let metrics_with_cache = CompletionMetrics {
            prompt_tokens: 10,
            completion_tokens: 20,
            prompt_eval_latency_ms: 100.0,
            completion_latency_ms: 200.0,
            cache_metrics: Some(CacheMetrics {
                cache_hit: true,
                strategy: Some("Hybrid".to_string()),
                tokens_skipped: Some(5),
                n_ctx: Some(2048),
                checkpoint_transition: Some("TurnStart".to_string()),
            }),
            ..Default::default()
        };
        let footer_with_cache = format_footer_metrics(&metrics_with_cache, Some("stop"), false);
        assert!(footer_with_cache.contains("5 cache"));

        // Test without cache_metrics (e.g., OpenAI provider)
        let metrics_no_cache = CompletionMetrics {
            prompt_tokens: 10,
            completion_tokens: 20,
            prompt_eval_latency_ms: 100.0,
            completion_latency_ms: 200.0,
            ..Default::default()
        };
        let footer_no_cache = format_footer_metrics(&metrics_no_cache, Some("stop"), false);
        assert!(footer_no_cache.contains("10 tok in"));
        assert!(footer_no_cache.contains("20 tok out"));
        assert!(!footer_no_cache.contains("ctx"));
        assert!(!footer_no_cache.contains("cache"));

        let cancelled_footer = format_footer_metrics(&metrics, None, true);
        assert_eq!("◼ Cancelled.", cancelled_footer);
    }
}
