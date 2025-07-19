use console::{Style, Term};
use std::io::Write;
use two_face::re_exports::syntect::{
    easy::HighlightLines,
    highlighting::{FontStyle, Style as SyntectStyle, Theme},
    parsing::SyntaxSet,
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

pub fn get_render_theme(theme_name: &str) -> Theme {
    // TODO: fix the theme to a terminal compatible one
    let theme_set = two_face::theme::extra();
    let theme_key = match theme_name {
        "dark" => two_face::theme::EmbeddedThemeName::Base16OceanDark,
        _ => two_face::theme::EmbeddedThemeName::Ansi,
    };
    // let theme_key = two_face::theme::EmbeddedThemeName::Ansi;
    theme_set.get(theme_key).clone()
    // let ansi_theme = include_str!("../../data/ansi.tmTheme");
    // let mut cursor = Cursor::new(ansi_theme.as_bytes());
    // ThemeSet::load_from_reader(&mut cursor).unwrap()
    // ThemeSet::get_theme("../../data/ansi.tmTheme").unwrap()
    // ThemeSet::load_defaults().themes["base16-ocean.dark"].clone()
}

pub struct TerminalRenderer<'a> {
    term: &'a mut Term,
    syntax_set: SyntaxSet,
    theme: &'a Theme,
    highlighter: HighlightLines<'a>,
}

impl<'a> TerminalRenderer<'a> {
    pub fn new(term: &'a mut Term, theme: &'a Theme) -> Self {
        let syntax_set = two_face::syntax::extra_newlines();
        let syntax = syntax_set.find_syntax_by_extension("md").unwrap();

        let highlighter = HighlightLines::new(syntax, theme);

        Self {
            term,
            syntax_set,
            theme,
            highlighter,
        }
    }

    pub fn clear(&mut self) {
        // Reset highlighter state for a new rendering session
        let syntax = self.syntax_set.find_syntax_by_extension("md").unwrap();
        self.highlighter = HighlightLines::new(syntax, self.theme);
    }

    pub fn test(&mut self) {
        let _ = self.render_markdown("# h1\n");
        let _ = self.render_markdown("# h1");
        let _ = self.render_markdown("# ");
    }

    pub fn render_markdown(&mut self, text: &str) -> Result<(), anyhow::Error> {
        // Text can be one of two states:
        // - continuation of previous line
        // - complete a previous line, start a newline
        let lines: Vec<&str> = LinesWithEndings::from(text).collect();
        // println!("\nin render: {text}, lines: {lines:?}");
        for line in lines {
            let ranges = self
                .highlighter
                .highlight_line(line, &self.syntax_set)
                .unwrap_or_else(|_| vec![(SyntectStyle::default(), line)]);

            let highlighted = self.to_ansi_terminal_escaped(&ranges[..]);
            // let highlighted = as_24_bit_terminal_escaped(&ranges[..], false);
            self.term.write_all(highlighted.as_bytes())?;
        }

        // `as_24_bit_terminal_escaped` doesn't reset the color
        // print!("\x1b[0m");
        Ok(())
    }

    fn to_ansi_terminal_escaped(&self, v: &[(SyntectStyle, &str)]) -> String {
        let mut s: String = String::new();

        for &(ref hl_style, text) in v.iter() {
            let mut style = Style::new().force_styling(true);
            style = match hl_style.font_style {
                FontStyle::BOLD => style.bold(),
                FontStyle::ITALIC => style.italic(),
                FontStyle::UNDERLINE => style.underlined(),
                _ => style,
            };

            // Apply the ansi colors for fg. We use the bg from terminal.
            // Ansi and base16-* themes set alpha to zero for fg, we'll translate it to console colors.
            // Update this to support other themes in future.
            // See https://github.com/sxyazi/yazi/blob/main/yazi-plugin/src/external/highlighter.rs for
            // inspiration.
            if hl_style.foreground.a == 0 {
                // println!("\n{hl_style:?} -- {text}");
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
                // println!("\nNot there: {hl_style:?} -- {text}");
            }

            s.push_str(style.apply_to(text).to_string().as_str());
        }

        s
    }
}
