//! Extensions for arey.
//! Support for tools, agents, workflows, and memory extensions.
use anyhow::{Context, Result, anyhow};
use std::sync::Arc;

use arey_core::{config::Config, registry::ToolRegistry, tools::Tool};

/// Retrieves all available tools from the configuration.
///
/// This function iterates through the tools defined in the provided configuration
/// and initializes each one based on its name. Currently, only the "search" tool
/// is supported.
///
/// # Arguments
///
/// * `config` - A reference to the configuration containing tool definitions.
///
/// # Returns
///
/// A `Result` containing a `ToolRegistry` with all available tools, or an error
/// if initialization fails or an unknown tool is encountered.
///
/// # Errors
///
/// Returns an error if:
/// - A tool fails to initialize (e.g., due to invalid configuration).
/// - An unknown tool name is found in the configuration.
pub fn get_tools(config: &Config) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    for (name, tool_config) in &config.tools {
        let tool: Arc<dyn Tool> = match name.as_str() {
            "search" => Arc::new(
                arey_tools_search::SearchTool::from_config(tool_config)
                    .with_context(|| format!("Failed to initialize tool: {name}"))?,
            ),
            _ => return Err(anyhow!("Unknown tool in config: {}", name)),
        };
        registry.register(tool)?;
    }

    Ok(registry)
}
