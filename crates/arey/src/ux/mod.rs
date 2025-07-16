mod presenter;
mod progress;
mod render;

pub use presenter::{ChatMessageType, format_footer_metrics, style_chat_text};
pub use progress::GenerationSpinner;
pub use render::{TerminalRenderer, get_render_theme};

use console::style;

pub fn present_error(error: anyhow::Error) {
    let error_text = style("ERROR:").red().bold();
    eprintln!("\n{error_text} {}", error);
}
