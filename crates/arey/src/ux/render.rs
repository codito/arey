use console::Style;
use std::io::{Cursor, Write};
use syntect::easy::HighlightLines;
use syntect::{
    highlighting::{
        FontStyle, HighlightState, Highlighter, Style as SyntectStyle, Theme, ThemeSet,
    },
    parsing::{ParseState, ScopeStack, SyntaxSet},
};

/// Returns a syntect theme for rendering.
pub fn get_theme(_theme_name: &str) -> Theme {
    let ansi_theme = include_str!("../../data/ansi.tmTheme");
    let mut cursor = Cursor::new(ansi_theme.as_bytes());
    ThemeSet::load_from_reader(&mut cursor).unwrap()
}

pub struct TerminalRenderer<'a> {
    term: &'a mut dyn Write,
    syntax_set: SyntaxSet,
    theme: &'a Theme,
    highlighter: HighlightLines<'a>,

    // For streaming rendering
    parser: ParseState,
    scope_stack: ScopeStack,
    line_buffer: String,
    stream_syntax_set: SyntaxSet,
    last_output_was_partial: bool,
}

impl<'a> TerminalRenderer<'a> {
    pub fn new(term: &'a mut dyn Write, theme: &'a Theme) -> Self {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let syntax = syntax_set.find_syntax_by_extension("md").unwrap();

        let highlighter = HighlightLines::new(syntax, theme);

        // For streaming
        let stream_syntax_set = SyntaxSet::load_defaults_nonewlines();
        let stream_syntax = stream_syntax_set.find_syntax_by_extension("md").unwrap();
        let parser = ParseState::new(stream_syntax);

        Self {
            term,
            syntax_set,
            theme,
            highlighter,
            parser,
            scope_stack: ScopeStack::new(),
            line_buffer: String::new(),
            stream_syntax_set,
            last_output_was_partial: false,
        }
    }

    pub fn clear(&mut self) {
        // Reset highlighter state for a new rendering session
        let syntax = self.syntax_set.find_syntax_by_extension("md").unwrap();
        self.highlighter = HighlightLines::new(syntax, self.theme);
    }

    pub fn render_markdown(&mut self, text: &str) -> Result<(), anyhow::Error> {
        self.line_buffer.push_str(text);
        let mut output = String::new();

        let highlighter = Highlighter::new(self.theme);

        // Process all complete lines in the buffer first.
        while let Some(i) = self.line_buffer.find('\n') {
            let line_to_process = self.line_buffer.drain(..=i).collect::<String>();
            // Exclude the newline character for parsing, but add it back to the output.
            let line_content = &line_to_process[..line_to_process.len() - 1];

            // Highlight the full line, advancing the persistent state.
            let ops = self
                .parser
                .parse_line(line_content, &self.stream_syntax_set)
                .unwrap();
            let mut highlight_state = HighlightState::new(&highlighter, self.scope_stack.clone());
            let ranges = syntect::highlighting::HighlightIterator::new(
                &mut highlight_state,
                &ops[..],
                line_content,
                &highlighter,
            )
            .collect::<Vec<_>>();
            self.scope_stack = highlight_state.path;

            output.push_str(&self.to_ansi_terminal_escaped(&ranges));
            output.push('\n');
        }

        // If there's a remaining partial line, highlight it without advancing state.
        if !self.line_buffer.is_empty() {
            // Use a temporary parser and highlighter state to not affect the real state.
            let mut temp_parser = self.parser.clone();
            let ops = temp_parser
                .parse_line(&self.line_buffer, &self.stream_syntax_set)
                .unwrap();
            let mut temp_highlighter = HighlightState::new(&highlighter, self.scope_stack.clone());
            let ranges = syntect::highlighting::HighlightIterator::new(
                &mut temp_highlighter,
                &ops[..],
                &self.line_buffer,
                &highlighter,
            )
            .collect::<Vec<_>>();
            output.push_str(&self.to_ansi_terminal_escaped(&ranges));
        }

        if output.is_empty() {
            return Ok(());
        }

        if self.last_output_was_partial {
            self.term.write_all(b"\r")?;
        }

        self.term.write_all(output.as_bytes())?;
        self.term.flush()?;

        self.last_output_was_partial = !output.ends_with('\n');

        Ok(())
    }

    fn to_ansi_terminal_escaped(&self, v: &[(SyntectStyle, &str)]) -> String {
        let mut s: String = String::new();
        let mut buffer = String::new();
        let mut current_style: Option<SyntectStyle> = None;

        let flush = |s: &mut String, buffer: &mut String, style: &Option<SyntectStyle>| {
            if buffer.is_empty() {
                return;
            }

            if let Some(style) = style {
                let mut console_style = Style::new().force_styling(true);

                // Apply color first
                if style.foreground.a == 0 {
                    console_style = match style.foreground.r {
                        0x00 => console_style.black(),
                        0x01 => console_style.red(),
                        0x02 => console_style.green(),
                        0x03 => console_style.yellow(),
                        0x04 => console_style.blue(),
                        0x05 => console_style.magenta(),
                        0x06 => console_style.cyan(),
                        0x07 => console_style.white(),
                        c => console_style.color256(c),
                    };
                }

                // Then apply font style
                console_style = match style.font_style {
                    FontStyle::BOLD => console_style.bold(),
                    FontStyle::ITALIC => console_style.italic(),
                    FontStyle::UNDERLINE => console_style.underlined(),
                    _ => console_style,
                };
                s.push_str(console_style.apply_to(&*buffer).to_string().as_str());
            } else {
                s.push_str(buffer);
            }
            buffer.clear();
        };

        for &(style, text) in v {
            if text.is_empty() {
                continue;
            }
            if Some(style) != current_style {
                flush(&mut s, &mut buffer, &current_style);
                current_style = Some(style);
            }
            buffer.push_str(text);
        }
        flush(&mut s, &mut buffer, &current_style);

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(chunks: &[&str]) -> String {
        let theme = get_theme("dark");
        let mut buffer = Vec::new();
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        for chunk in chunks {
            renderer.render_markdown(chunk).unwrap();
        }

        String::from_utf8(buffer).unwrap()
    }

    #[test]
    fn test_render_full_line() {
        let output = render(&["# Hello\n"]);
        assert_eq!(
            output,
            "\u{1b}[34m\u{1b}[1m#\u{1b}[0m \u{1b}[34m\u{1b}[1mHello\u{1b}[0m\n"
        );
    }

    #[test]
    fn test_render_partial_line() {
        let output = render(&["# He", "llo\n"]);
        let part1 = "\u{1b}[34m\u{1b}[1m#\u{1b}[0m \u{1b}[34m\u{1b}[1mHe\u{1b}[0m";
        let part2 = "\u{1b}[34m\u{1b}[1m#\u{1b}[0m \u{1b}[34m\u{1b}[1mHello\u{1b}[0m\n";
        assert_eq!(output, format!("{part1}\r{part2}"));
    }

    #[test]
    fn test_render_char_by_char() {
        let chars: Vec<String> = "`a`\n".chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(|s| s.as_str()).collect();
        let output = render(&refs);

        let expected = [
            "\u{1b}[32m`\u{1b}[0m",
            "\r\u{1b}[32m`a\u{1b}[0m",
            "\r\u{1b}[32m`a`\u{1b}[0m",
            "\r\u{1b}[32m`a`\u{1b}[0m\n",
        ]
        .join("");
        assert!(output.ends_with(&expected));
    }

    #[test]
    fn test_render_multiple_newlines() {
        let output = render(&["Hello\n", "\nWorld\n"]);
        assert!(output.ends_with("Hello\n\nWorld\n"));
    }

    #[test]
    fn test_render_split_markdown() {
        let output = render(&["*ite", "m*\n"]);
        let part1 = "\u{1b}[35m\u{1b}[3m*ite\u{1b}[0m";
        let part2 = "\u{1b}[35m\u{1b}[3m*item*\u{1b}[0m\n";
        assert!(output.ends_with(&format!("{part1}\r{part2}")));
    }

    #[test]
    #[ignore = "not supported yet"]
    fn test_render_code_block() {
        let output = render(&["```rust\n", "fn main() {}\n", "```\n"]);
        let line1 = "```\u{1b}[33mrust\u{1b}[0m\n";
        let line2 = "\u{1b}[35mfn\u{1b}[0m \u{1b}[34mmain\u{1b}[0m() {}\n";
        let line3 = "```\n";
        assert_eq!(output, format!("{line1}{line2}{line3}"));
    }
}
