//! GGUF model provider for llama.cpp based inference.

pub use model::GgufBaseModel;

mod checkpoint;
mod model;
mod template;

pub use template::{ToolCallParser, apply_chat_template};
