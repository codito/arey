// mod console;
mod progress;
mod render;
// mod presenter;

use arey_core::error::AnyError;
use console::style;
use std::fmt::Display;

// pub use console::*;
// pub use presenter::*;
pub use progress::*;
pub use render::*;

pub enum TextStyle {
    Command,
    Agent,
    Workflow,
    Info,
}

pub fn present_error(error: impl AnyError) {
    let error_text = style("ERROR:").red().bold();
    eprintln!("\n{error_text} {}", error);
}

pub fn style_text(text: impl Display, style: TextStyle) -> String {
    match style {
        TextStyle::Command => style(text).bold().yellow().to_string(),
        TextStyle::Agent => style(text).bold().green().to_string(),
        TextStyle::Workflow => style(text).bold().magenta().to_string(),
        TextStyle::Info => style(text).dim().to_string(),
    }
}
