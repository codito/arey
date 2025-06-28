use arey_core::completion::CompletionMetrics;
use console::{Style, StyledObject, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use two_face::re_exports::syntect::{
    easy::HighlightLines,
    highlighting::{Style as SyntectStyle, Theme, ThemeSet},
    parsing::SyntaxSet,
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Prompt,
    // User,
    // AI,
    Footer,
    Error,
}

pub fn style_text(text: &str, style: MessageType) -> StyledObject<&str> {
    let style_obj = match style {
        MessageType::Prompt => Style::new().blue().bold(),
        // MessageType::User => Style::new().blue(),
        // MessageType::AI => Style::new().white().bright(),
        MessageType::Footer => Style::new().white().dim(),
        MessageType::Error => Style::new().red().bold(),
    };
    style_obj.apply_to(text)
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

    style_text(&footer, MessageType::Footer).to_string()
}

pub fn get_render_theme() -> Theme {
    let theme_set = two_face::theme::extra();
    theme_set
        .get(two_face::theme::EmbeddedThemeName::Base16OceanLight)
        .clone()
}

pub struct MarkdownRenderer<'a> {
    term: &'a mut Term,
    syntax_set: SyntaxSet,
    theme: &'a Theme,
    highlighter: HighlightLines<'a>,

    // Store the current incomplete line buffer
    line_buffer: String,
    // Stores previously rendered complete lines (already highlighted and escaped)
    prev_rendered_lines: Vec<String>,
    // Cursor position where rendering started
    start_pos: Option<(usize, usize)>,
    // Number of lines previously drawn by this renderer
    prev_line_count: usize,
}

impl<'a> MarkdownRenderer<'a> {
    pub fn new(term: &'a mut Term, theme: &'a Theme) -> Self {
        let syntax_set = two_face::syntax::extra_newlines();
        let syntax = syntax_set.find_syntax_by_extension("md").unwrap();
        let highlighter = HighlightLines::<'a>::new(syntax, theme);

        Self {
            term,
            syntax_set,
            theme,
            highlighter,
            line_buffer: String::new(),
            prev_rendered_lines: Vec::new(),
            start_pos: None,
            prev_line_count: 0,
        }
    }

    pub fn clear(&mut self) {
        // This clears the internal buffer and resets position tracking for a new render.
        // It does NOT clear the screen area. Screen clearing is handled within `render`.
        self.line_buffer.clear();
        self.prev_rendered_lines.clear();
        self.start_pos = None;
        self.prev_line_count = 0;

        // Reset highlighter state for a new rendering session
        let syntax = self.syntax_set.find_syntax_by_extension("md").unwrap();
        self.highlighter = HighlightLines::new(syntax, self.theme);
    }

    pub fn render(&mut self, text: &str) -> Result<(), anyhow::Error> {
        // Text can be one of two states:
        // - continuation of previous line
        // - complete a previous line, start a newline
        let lines: Vec<&str> = LinesWithEndings::from(text).collect();
        for line in lines {
            let ranges = self
                .highlighter
                .highlight_line(line, &self.syntax_set)
                .unwrap_or_else(|_| vec![(SyntectStyle::default(), line)]);

            let highlighted = as_24_bit_terminal_escaped(&ranges[..], true);
            self.term.write_all(highlighted.as_bytes())?;
        }
        Ok(())
    }
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
