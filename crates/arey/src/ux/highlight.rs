use std::io::Cursor;

use console::Style;
use syntect::{
    highlighting::{
        FontStyle, HighlightState, Highlighter, Style as SyntectStyle, Theme, ThemeSet,
    },
    parsing::{ParseState, ScopeStack, SyntaxSet},
};

/// Returns a syntect theme for rendering.
pub fn get_theme(_theme_name: &str) -> Theme {
    // TODO: fix the theme to a terminal compatible one
    // let theme_set = ThemeSet::new();
    // let theme_key = match theme_name {
    //     "dark" => two_face::theme::EmbeddedThemeName::Base16OceanDark,
    //     _ => two_face::theme::EmbeddedThemeName::Ansi,
    // };
    // theme_set.get(theme_key).clone()
    let ansi_theme = include_str!("../../data/ansi.tmTheme");
    let mut cursor = Cursor::new(ansi_theme.as_bytes());
    ThemeSet::load_from_reader(&mut cursor).unwrap()
}

/// A stateful markdown highlighter for streaming terminal output.
///
/// This highlighter manages `syntect`'s `ParseState` and `ScopeStack`
/// to provide highlighting for streaming content, including partial lines.
pub struct MarkdownHighlighter<'a> {
    syntax_set: SyntaxSet,
    theme: &'a Theme,

    // The persistent state for highlighting across multiple lines.
    parser: ParseState,
    scope_stack: ScopeStack,

    // Buffer for the current, incomplete line of text.
    line_buffer: String,
}

impl<'a> MarkdownHighlighter<'a> {
    /// Creates a new markdown highlighter with a given theme.
    pub fn new(theme: &'a Theme) -> Self {
        let syntax_set = two_face::syntax::extra_newlines();
        let syntax = syntax_set.find_syntax_by_extension("md").unwrap();
        let parser = ParseState::new(syntax);

        Self {
            syntax_set,
            theme,
            parser,
            scope_stack: ScopeStack::new(),
            line_buffer: String::new(),
        }
    }

    /// Resets the highlighter state for a new highlighting session.
    pub fn new_session(&mut self) {
        let syntax = self.syntax_set.find_syntax_by_extension("md").unwrap();
        self.parser = ParseState::new(syntax);
        self.scope_stack = ScopeStack::new();
        self.line_buffer.clear();
    }

    /// Highlights a chunk of text, supporting streaming of partial lines.
    ///
    /// This method processes the incoming text chunk, highlighting and returning
    /// any complete lines found. It also highlights the current partial line
    /// without permanently advancing the highlighter's state, allowing for
    /// continuous updates as more text streams in.
    ///
    /// The caller is responsible for managing the terminal display, such as
    /// overwriting a previously rendered partial line with the new version from
    /// this function's output.
    pub fn highlight(&mut self, text: &str) -> String {
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
                .parse_line(line_content, &self.syntax_set)
                .unwrap();
            let mut highlight_state = HighlightState::new(&highlighter, self.scope_stack.clone());
            let ranges = syntect::highlighting::HighlightIterator::new(
                &mut highlight_state,
                &ops[..],
                line_content,
                &self.syntax_set,
            )
            .collect::<Vec<_>>();
            self.scope_stack = highlight_state.path;

            output.push_str(&to_ansi_terminal_escaped(&ranges));
            output.push('\n');
        }

        // If there's a remaining partial line, highlight it without advancing state.
        if !self.line_buffer.is_empty() {
            // Use a temporary parser and highlighter state to not affect the real state.
            let mut temp_parser = self.parser.clone();
            let ops = temp_parser
                .parse_line(&self.line_buffer, &self.syntax_set)
                .unwrap();
            let mut temp_highlighter = HighlightState::new(&highlighter, self.scope_stack.clone());
            let ranges = syntect::highlighting::HighlightIterator::new(
                &mut temp_highlighter,
                &ops[..],
                &self.line_buffer,
                &self.syntax_set,
            )
            .collect::<Vec<_>>();
            output.push_str(&to_ansi_terminal_escaped(&ranges));
        }

        output
    }
}

/// Converts syntect's styled ranges to an ANSI-escaped string for terminals.
fn to_ansi_terminal_escaped(v: &[(SyntectStyle, &str)]) -> String {
    let mut s: String = String::new();

    for &(ref hl_style, text) in v.iter() {
        let mut style = Style::new().force_styling(true);
        style = match hl_style.font_style {
            FontStyle::BOLD => style.bold(),
            FontStyle::ITALIC => style.italic(),
            FontStyle::UNDERLINE => style.underlined(),
            _ => style,
        };

        // This handles themes that use special ANSI color encoding.
        if hl_style.foreground.a == 0 {
            style = match hl_style.foreground.r {
                0x00 => style.black(),
                0x01 => style.red(),
                0x02 => style.green(),
                0x03 => style.yellow(),
                0x04 => style.blue(),
                0x05 => style.magenta(),
                0x06 => style.cyan(),
                0x07 => style.white(),
                c => style.color256(c),
            };
        } else {
            // Fallback for themes with standard RGB colors (optional).
        }

        s.push_str(style.apply_to(text).to_string().as_str());
    }

    s
}
