mod presenter;
mod progress;
mod render;

pub use presenter::{ChatMessageType, format_footer_metrics, style_chat_text};
pub use progress::GenerationSpinner;
pub use render::*;

use console::style;

/// Prints a formatted error message to stderr.
pub fn present_error(error: anyhow::Error) {
    let error_text = style("ERROR:").red().bold();
    eprintln!("\n{error_text} {error}");
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for `present_error`. This would involve
    // capturing stderr to verify the output.
}
