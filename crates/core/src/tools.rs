use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool execution failed: {0}")]
    ExecutionError(String),
}

/// Represents a tool call requested by the model.
/// A tool call may or may not resolve into a `Tool`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolCall {
    pub id: String, // To uniquely identify the call
    pub name: String,
    pub arguments: String,
}

/// The result of a tool execution, to be sent back to the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
    pub call: ToolCall,
    pub output: Value,
}

/// Specification for describing tools to models
#[derive(Serialize, Debug, Clone)]
pub struct ToolSpec {
    /// Currently only "function" type is supported
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function specification details
    pub function: FunctionSpec,
}

/// Specification format for functions (tools as functions)
#[derive(Serialize, Debug, Clone)]
pub struct FunctionSpec {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    pub description: String,
    /// JSON schema representing function parameters
    pub parameters: Value,
}

/// Conversion from Tool trait object to ToolSpec
impl From<&dyn Tool> for ToolSpec {
    fn from(tool: &dyn Tool) -> Self {
        ToolSpec {
            tool_type: "function".to_string(),
            function: FunctionSpec {
                name: tool.name(),
                description: tool.description(),
                parameters: tool.parameters(),
            },
        }
    }
}

/// A trait for defining tools that can be used by the model.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The name of the tool, used to identify it.
    fn name(&self) -> String;

    /// A description of what the tool does, for the model to understand its purpose.
    fn description(&self) -> String;

    /// The parameters the tool accepts, as a JSON schema Value.
    fn parameters(&self) -> Value;

    /// Execute the tool with the given arguments.
    async fn execute(&self, arguments: &Value) -> Result<Value, ToolError>;
}
