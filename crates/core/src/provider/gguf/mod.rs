//! GGUF model provider for llama.cpp based inference.

pub use model::GgufBaseModel;

mod checkpoint;
mod model;
mod system;
mod template;
mod tool;

pub use system::{
    ComputeParams, ConfigOverrides, compute_params, detect_system_info, flash_attention_policy,
};
pub use template::apply_chat_template;
pub use tool::{TemplatePatterns, ToolCallParser, get_tool_call_regexes};
