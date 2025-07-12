use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionTool, ChatCompletionToolType, CompletionUsage,
    FinishReason, FunctionCall, FunctionCallStream,
};
use serde::Deserialize;

use crate::tools::{Tool, ToolCall};

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionStreamResponse {
    pub(super) choices: Vec<ChatCompletionStreamChoice>,
    pub(super) usage: Option<CompletionUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionStreamChoice {
    pub(super) delta: ChatCompletionDelta,
    pub(super) finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionDelta {
    pub(super) content: Option<String>,
    pub(super) tool_calls: Option<Vec<ChatCompletionToolCallChunk>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionToolCallChunk {
    // Use a default value for tool call index (Gemini bug).
    // See https://discuss.ai.google.dev/t/tool-calling-with-openai-api-not-working/60140/5
    #[serde(default)]
    pub(super) index: u32,
    pub(super) id: Option<String>,
    pub(super) function: Option<FunctionCallStream>,
}

impl dyn Tool {
    /// Converts a tool into `ChatCompletionTool` openai format.
    pub fn to_openai_tool(&self) -> ChatCompletionTool {
        ChatCompletionTool {
            r#type: async_openai::types::ChatCompletionToolType::Function,
            function: async_openai::types::FunctionObject {
                name: self.name(),
                description: Some(self.description()),
                parameters: Some(self.parameters()),
                strict: None,
            },
        }
    }
}

impl From<ChatCompletionToolCallChunk> for ToolCall {
    fn from(value: ChatCompletionToolCallChunk) -> Self {
        ToolCall {
            id: value.id.unwrap(),
            name: value
                .function
                .as_ref()
                .and_then(|f| f.name.clone())
                .unwrap(),
            arguments: value.function.and_then(|f| f.arguments).unwrap(),
        }
    }
}

impl From<ToolCall> for ChatCompletionMessageToolCall {
    fn from(val: ToolCall) -> Self {
        // Use name as id if it is not available (Gemini bug)
        // See https://discuss.ai.google.dev/t/tool-calling-with-openai-api-not-working/60140/5
        let id = if val.id.is_empty() {
            val.name.clone()
        } else {
            val.id
        };
        ChatCompletionMessageToolCall {
            id,
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: val.name,
                arguments: val.arguments,
            },
        }
    }
}
