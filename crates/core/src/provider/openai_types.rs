use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionStreamResponse {
    pub(super) choices: Vec<ChatCompletionStreamChoice>,
    pub(super) usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionStreamChoice {
    pub(super) delta: Delta,
    pub(super) finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Delta {
    pub(super) content: Option<String>,
    pub(super) tool_calls: Option<Vec<ToolCallChunk>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ToolCallChunk {
    #[serde(default)]
    pub(super) index: u32,
    pub(super) id: Option<String>,
    pub(super) function: Option<FunctionCallChunk>,
}

#[derive(Debug, Deserialize)]
pub(super) struct FunctionCallChunk {
    pub(super) name: Option<String>,
    pub(super) arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Usage {
    pub(super) prompt_tokens: u32,
    pub(super) completion_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}
