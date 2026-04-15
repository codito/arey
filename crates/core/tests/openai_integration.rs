use arey_core::tools::{Tool, ToolError};
use arey_core::{
    completion::{CancellationToken, ChatMessage, Completion, SenderType},
    get_completion_llm,
    model::{ModelConfig, ModelProvider},
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use serde_json::json;
use serde_yaml::Value as YamlValue;
use std::{collections::HashMap, env, sync::Arc};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> String {
        "get_current_weather".to_string()
    }

    fn description(&self) -> String {
        "Gets the current weather for a given location".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        })
    }

    async fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value, ToolError> {
        let location = arguments["location"]
            .as_str()
            .ok_or_else(|| ToolError::ExecutionError("Missing location parameter".to_string()))?;

        Ok(json!({
            "location": location,
            "temp_C": 20,
            "weatherDesc": "partly cloudy",
            "humidity": 65,
            "precipMM": 0
        }))
    }
}

async fn test_tool_calling(
    model_name: &str,
    base_url: &str,
    api_key_env: &str,
    n_ctx: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CI").is_ok() {
        eprintln!("Skipping {} tool calling test on CI", model_name);
        return Ok(());
    }

    let api_key = match env::var(api_key_env) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "Skipping {} tool calling test: {} not set",
                model_name, api_key_env
            );
            return Ok(());
        }
    };

    let mut settings = HashMap::from([
        (
            "base_url".to_string(),
            YamlValue::String(base_url.to_string()),
        ),
        ("api_key".to_string(), YamlValue::String(api_key)),
    ]);
    if let Some(ctx) = n_ctx {
        settings.insert("n_ctx".to_string(), YamlValue::Number(ctx.into()));
    }

    let config = ModelConfig {
        key: model_name.to_string(),
        name: model_name.to_string(),
        provider: ModelProvider::Openai,
        settings,
    };

    let model = get_completion_llm(config)?;
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(WeatherTool)];

    let messages = vec![ChatMessage {
        text: "What's the current weather in London?".to_string(),
        sender: SenderType::User,
        ..Default::default()
    }];

    let cancel_token = CancellationToken::new();
    let mut assistant_content = String::new();
    let mut tool_calls = Vec::new();

    let mut stream = model
        .complete(&messages, Some(&tools), &HashMap::new(), cancel_token)
        .await;

    let stream_result: Result<(), Box<dyn std::error::Error>> = async {
        while let Some(chunk) = stream.next().await {
            match chunk? {
                Completion::Response(response) => {
                    assistant_content.push_str(&response.text);
                    if let Some(calls) = response.tool_calls {
                        tool_calls.extend(calls);
                    }
                }
                Completion::Metrics(_) => {}
            }
        }
        Ok(())
    }
    .await;

    if let Err(e) = &stream_result {
        let err_str = e.to_string();
        if err_str.contains("too_many_requests")
            || err_str.contains("rate_limit")
            || err_str.contains("queue_exceeded")
            || err_str.contains("rate-limited")
            || err_str.contains("429")
        {
            eprintln!("Skipping {} tool calling test: rate limited", model_name);
            return Ok(());
        }
        return stream_result;
    }

    assert!(
        !tool_calls.is_empty(),
        "Expected tool call but got none. Response: {}",
        assistant_content
    );

    let tool_call = &tool_calls[0];
    assert_eq!(
        tool_call.name, "get_current_weather",
        "Expected get_current_weather tool call"
    );

    let tool = tools
        .iter()
        .find(|t| t.name() == tool_call.name)
        .expect("Tool not found");
    let args: serde_json::Value =
        serde_json::from_str(&tool_call.arguments).expect("Failed to parse tool arguments");
    let output = tool.execute(&args).await?;

    let temp = output["temp_C"].as_f64().expect("Missing temp in output");
    assert!(
        temp > -50.0 && temp < 60.0,
        "Temperature out of realistic range"
    );

    println!(
        "Tool call test passed for {}: {} -> {:?}",
        model_name, tool_call.name, output
    );

    Ok(())
}

#[tokio::test]
#[ignore]
async fn gemini_2_5_tool_calling() {
    test_tool_calling(
        "gemini-2.5-flash",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        Some(1048576),
    )
    .await
    .expect("gemini-2.5-flash tool calling test failed");
}

#[tokio::test]
#[ignore]
async fn gemini_3_tool_calling() {
    test_tool_calling(
        "gemini-3-flash-preview",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        Some(1048576),
    )
    .await
    .expect("gemini-3-flash-preview tool calling test failed");
}

#[tokio::test]
#[ignore]
async fn cerebras_tool_calling() {
    test_tool_calling(
        "qwen-3-235b-a22b-instruct-2507",
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
        Some(65536),
    )
    .await
    .expect("cerebras tool calling test failed");
}

#[tokio::test]
#[ignore]
async fn openrouter_tool_calling() {
    test_tool_calling(
        "openrouter/elephant-alpha",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        Some(2621024),
    )
    .await
    .expect("openrouter tool calling test failed");
}
