use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
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

/// Conversion from Arc<dyn Tool> to ToolSpec
impl From<Arc<dyn Tool>> for ToolSpec {
    fn from(tool: Arc<dyn Tool>) -> Self {
        ToolSpec::from(&*tool)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> String {
            "test_tool".to_string()
        }

        fn description(&self) -> String {
            "A test tool for unit testing".to_string()
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Test input parameter"
                    }
                },
                "required": ["input"]
            })
        }

        async fn execute(&self, _arguments: &Value) -> Result<Value, ToolError> {
            Ok(json!({"status": "success"}))
        }
    }

    #[test]
    fn test_tool_spec_conversion() {
        let mock_tool = MockTool;
        let mock_tool_arc: Arc<dyn Tool> = Arc::new(mock_tool);
        let tool_spec = ToolSpec::from(&*mock_tool_arc);

        assert_eq!(tool_spec.tool_type, "function");
        assert_eq!(tool_spec.function.name, "test_tool");
        assert_eq!(
            tool_spec.function.description,
            "A test tool for unit testing"
        );

        let expected_params = json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Test input parameter"
                }
            },
            "required": ["input"]
        });
        assert_eq!(tool_spec.function.parameters, expected_params);
    }

    #[test]
    fn test_tool_spec_serialization() {
        let mock_tool = MockTool;
        let mock_tool_arc: Arc<dyn Tool> = Arc::new(mock_tool);
        let tool_spec = ToolSpec::from(&*mock_tool_arc);

        let serialized = serde_json::to_value(&tool_spec).unwrap();
        let expected = json!({
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool for unit testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Test input parameter"
                        }
                    },
                    "required": ["input"]
                }
            }
        });

        assert_eq!(serialized, expected);
    }
}
