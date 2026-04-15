use crate::completion::{ChatMessage, SenderType};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;
use tracing::debug;

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool execution failed: {0}")]
    ExecutionError(String),
}

pub struct ToolExecutor;

impl ToolExecutor {
    pub async fn execute(call: &ToolCall, tool: &dyn Tool) -> Result<ChatMessage, ToolError> {
        let args = Self::normalize_arguments(&call.arguments)?;
        debug!("Executing tool: {} with arguments: {}", call.name, args);
        let mut output = tool
            .execute(&args)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        // Clean control characters from tool output
        Self::clean_value(&mut output);

        Ok(Self::create_result_message(call, output))
    }

    /// Recursively removes control characters from JSON values
    fn clean_value(value: &mut Value) {
        match value {
            Value::String(s) => {
                let cleaned = s.replace(|c: char| c.is_control(), "");
                *s = cleaned;
            }
            Value::Array(a) => a.iter_mut().for_each(Self::clean_value),
            Value::Object(m) => m.values_mut().for_each(Self::clean_value),
            _ => {}
        }
    }

    pub fn normalize_arguments(arguments: &str) -> Result<Value, ToolError> {
        let args = match serde_json::from_str::<Value>(arguments) {
            Ok(value) => value,
            Err(_first_error) => match serde_json::from_str::<Value>(arguments) {
                Ok(value) => value,
                Err(_) => {
                    serde_json::json!({ "input": arguments })
                }
            },
        };

        let args = match &args {
            Value::String(s) => match serde_json::from_str::<Value>(s) {
                Ok(parsed_value) => parsed_value,
                Err(_) => args,
            },
            _ => args,
        };

        Ok(args)
    }

    pub fn create_result_message(call: &ToolCall, output: Value) -> ChatMessage {
        let mut call_clone = call.clone();
        if call.id.is_empty() {
            call_clone.id = format!("call_{}", rand_id());
        }

        let tool_result = ToolResult {
            call: call_clone,
            output,
        };

        ChatMessage {
            sender: SenderType::Tool,
            text: serde_json::to_string(&tool_result).unwrap_or_default(),
            ..Default::default()
        }
    }

    pub fn create_error_message(call: &ToolCall, error: &ToolError) -> ChatMessage {
        let mut call_clone = call.clone();
        if call.id.is_empty() {
            call_clone.id = format!("call_{}", rand_id());
        }

        let tool_result = ToolResult {
            call: call_clone,
            output: serde_json::json!({ "error": error.to_string() }),
        };

        ChatMessage {
            sender: SenderType::Tool,
            text: serde_json::to_string(&tool_result).unwrap_or_default(),
            ..Default::default()
        }
    }
}

fn rand_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}_{}", duration.as_nanos(), std::process::id())
}

/// Represents a tool call requested by the model.
/// A tool call may or may not resolve into a `Tool`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub extra_content: Option<serde_json::Value>,
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
mod spec_tests {
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

#[cfg(test)]
mod executor_tests {
    use super::*;
    use crate::completion::SenderType;
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::{Value, json};

    #[derive(Clone)]
    struct TestTool;

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> String {
            "test_tool".to_string()
        }

        fn description(&self) -> String {
            "A test tool".to_string()
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _args: &Value) -> Result<Value, ToolError> {
            Ok(Value::String("mock tool output".to_string()))
        }
    }

    #[test]
    fn test_normalize_arguments_direct_json() {
        let args = r#"{"key": "value"}"#;
        let result = ToolExecutor::normalize_arguments(args);

        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value.get("key").unwrap().as_str(), Some("value"));
    }

    #[test]
    fn test_normalize_arguments_plain_string() {
        let args = "plain string";
        let result = ToolExecutor::normalize_arguments(args);

        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value.get("input").unwrap().as_str(), Some("plain string"));
    }

    #[test]
    fn test_create_result_message() {
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
            ..Default::default()
        };

        let output = json!({"result": "success"});
        let msg = ToolExecutor::create_result_message(&call, output);

        assert_eq!(msg.sender, SenderType::Tool);
        assert!(msg.text.contains("call_1"));
    }

    #[test]
    fn test_create_error_message() {
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
            ..Default::default()
        };

        let error = ToolError::ExecutionError("Tool failed".to_string());
        let msg = ToolExecutor::create_error_message(&call, &error);

        assert_eq!(msg.sender, SenderType::Tool);
        assert!(msg.text.contains("error"));
    }

    #[test]
    fn test_create_result_message_preserves_extra_content() {
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
            extra_content: Some(json!({"google": {"thought_signature": "sig_abc"}})),
        };

        let output = json!({"result": "success"});
        let msg = ToolExecutor::create_result_message(&call, output);

        assert_eq!(msg.sender, SenderType::Tool);
        let tool_result: ToolResult = serde_json::from_str(&msg.text).unwrap();
        assert_eq!(
            tool_result.call.extra_content,
            Some(json!({"google": {"thought_signature": "sig_abc"}}))
        );
    }

    #[tokio::test]
    async fn test_execute() -> Result<()> {
        let tool = TestTool;
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{key: 10}".to_string(),
            ..Default::default()
        };

        let msg = ToolExecutor::execute(&call, &tool).await?;

        assert_eq!(msg.sender, SenderType::Tool);
        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;
        assert_eq!(tool_result.call.id, "call_1");
        assert_eq!(tool_result.output, json!("mock tool output"));

        Ok(())
    }

    #[tokio::test]
    async fn test_execute_with_stringified_json_argument() -> Result<()> {
        struct ArgRecorder;
        #[async_trait]
        impl Tool for ArgRecorder {
            fn name(&self) -> String {
                "arg_recorder".to_string()
            }

            fn description(&self) -> String {
                "Records arguments".to_string()
            }

            fn parameters(&self) -> Value {
                json!({})
            }

            async fn execute(&self, args: &Value) -> std::result::Result<Value, ToolError> {
                Ok(args.clone())
            }
        }

        let tool = ArgRecorder;
        // The arguments are a string that itself is a JSON object
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "arg_recorder".to_string(),
            arguments: r#""{\"arg\":42}""#.to_string(),
            ..Default::default()
        };

        let result = ToolExecutor::execute(&call, &tool).await?;

        // The tool should have received the parsed JSON object: {"arg":42}
        let tool_result: ToolResult = serde_json::from_str(&result.text)?;
        assert_eq!(tool_result.output, json!({"arg":42}));

        Ok(())
    }

    #[tokio::test]
    async fn test_execute_with_non_json_string() -> Result<()> {
        struct ArgRecorder;
        #[async_trait]
        impl Tool for ArgRecorder {
            fn name(&self) -> String {
                "arg_recorder".to_string()
            }

            fn description(&self) -> String {
                "Records arguments".to_string()
            }

            fn parameters(&self) -> Value {
                json!({})
            }

            async fn execute(&self, args: &Value) -> std::result::Result<Value, ToolError> {
                Ok(args.clone())
            }
        }

        let tool = ArgRecorder;
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "arg_recorder".to_string(),
            arguments: "plain string".to_string(),
            ..Default::default()
        };

        let result = ToolExecutor::execute(&call, &tool).await?;

        // We expect the tool to have received: {"input": "plain string"}
        let tool_result: ToolResult = serde_json::from_str(&result.text)?;
        assert_eq!(tool_result.output, json!({ "input": "plain string" }));

        Ok(())
    }

    #[tokio::test]
    async fn test_execute_cleans_control_characters() -> Result<()> {
        #[derive(Debug)]
        struct ControlCharTool;
        #[async_trait]
        impl Tool for ControlCharTool {
            fn name(&self) -> String {
                "control_tool".to_string()
            }

            fn description(&self) -> String {
                "Tool with control characters".to_string()
            }

            fn parameters(&self) -> Value {
                json!({})
            }

            async fn execute(&self, _args: &Value) -> std::result::Result<Value, ToolError> {
                // Return value containing control characters
                Ok(json!({
                    "key": "value with \u{0001} control \u{001F} characters"
                }))
            }
        }

        let tool = ControlCharTool;
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "control_tool".to_string(),
            arguments: "{}".to_string(),
            ..Default::default()
        };

        let result = ToolExecutor::execute(&call, &tool).await?;

        // Check that control characters were removed
        let tool_result: ToolResult = serde_json::from_str(result.text.as_str())?;
        let output_str = tool_result.output.get("key").unwrap().as_str().unwrap();
        assert_eq!(output_str, "value with  control  characters");
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_mismatched_name() {
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "different_tool".to_string(),
            arguments: "{}".to_string(),
            ..Default::default()
        };

        let tool = TestTool;
        let result = ToolExecutor::execute(&call, &tool).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execute_preserves_extra_content() -> Result<()> {
        let tool = TestTool;
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
            extra_content: Some(json!({"google": {"thought_signature": "test_sig_123"}})),
        };

        let msg = ToolExecutor::execute(&call, &tool).await?;

        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;
        assert_eq!(
            tool_result.call.extra_content,
            Some(json!({"google": {"thought_signature": "test_sig_123"}}))
        );

        Ok(())
    }
}
