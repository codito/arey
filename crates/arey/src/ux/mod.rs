mod presenter;
mod progress;
mod render;

pub use render::{TerminalRenderer, get_render_theme};

use console::style;

pub fn present_error(error: anyhow::Error) {
    let error_text = style("ERROR:").red().bold();
    eprintln!("\n{error_text} {}", error);
}
