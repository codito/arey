use anyhow::Result;
use arey_core::completion::{CancellationToken, ChatMessage, Completion, SenderType};
use arey_core::model::ModelConfig;
use arey_core::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> String {
        "get_current_weather".to_string()
    }

    fn description(&self) -> String {
        "Gets the current weather for a given location".to_string()
    }

    fn parameters(&self) -> Value {
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

    async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let location = arguments["location"]
            .as_str()
            .ok_or_else(|| ToolError::ExecutionError("Missing location parameter".to_string()))?;

        // Extract relevant fields
        Ok(json!({
            "location": location,
            "temp_C": 28,
            "weatherDesc": "sunny",
            "humidity": 54,
            "precipMM": 0
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Read API key from environment
    // let api_key = env::var("GROQ_API_KEY").expect("GROQ_API_KEY environment variable not set");

    // Configure model
    // let config = ModelConfig {
    //     name: "qwen-qwq-32b".to_string(),
    //     provider: arey_core::model::ModelProvider::Openai,
    //     settings: HashMap::from([
    //         (
    //             "base_url".to_string(),
    //             YamlValue::String("https://api.groq.com/openai/v1".to_string()),
    //         ),
    //         ("api_key".to_string(), YamlValue::String(api_key)),
    //     ]),
    // };

    let api_key = env::var("GEMINI_API_KEY").expect("GROQ_API_KEY environment variable not set");
    let config = ModelConfig {
        name: "gemini-2.5-flash".to_string(),
        provider: arey_core::model::ModelProvider::Openai,
        settings: HashMap::from([
            (
                "base_url".to_string(),
                YamlValue::String(
                    "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
                ),
            ),
            ("api_key".to_string(), YamlValue::String(api_key)),
        ]),
    };

    // Instantiate model and tools
    let mut model = arey_core::get_completion_llm(config)?;
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(WeatherTool)];
    model.load("").await?;

    let mut messages = vec![ChatMessage {
        text: "What's the current weather in London?".to_string(),
        sender: SenderType::User,
    }];
    println!("> {}", messages.last().unwrap().text);

    // Run completion
    let cancel_token = CancellationToken::new();
    let mut assistant_content = String::new();
    let mut tool_calls = Vec::new();
    {
        let mut stream = model
            .complete(
                &messages,
                Some(&tools),
                &HashMap::new(),
                cancel_token.clone(),
            )
            .await;

        // Process responses
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
    }

    // Add assistant message to history.
    messages.push(ChatMessage {
        text: assistant_content.clone(),
        sender: SenderType::Assistant,
    });

    if !tool_calls.is_empty() {
        if !assistant_content.is_empty() {
            println!("\nAssistant: {assistant_content}");
        }

        let mut tool_result_messages = Vec::new();
        for call in tool_calls {
            println!("\nTool call: {}({})", call.name, call.arguments);
            let tool = tools
                .iter()
                .find(|t| t.name() == call.name)
                .expect("Tool not found");
            let output = tool.execute(&call.arguments).await?;
            println!("Tool output: {output}");

            // The provider needs to know which tool call this result is for.
            // We assume it can parse this from the text of a Tool message.
            tool_result_messages.push(ChatMessage {
                sender: SenderType::Tool,
                text: serde_json::to_string(&ToolResult { call, output }).unwrap(),
            });
        }
        messages.extend(tool_result_messages);

        println!("\nSending tool results to model for final answer...");

        let mut stream = model
            .complete(&messages, Some(&tools), &HashMap::new(), cancel_token)
            .await;

        print!("\nFinal Assistant Response:\n");
        while let Some(chunk) = stream.next().await {
            match chunk? {
                Completion::Response(response) => {
                    print!("{}", response.text);
                }
                Completion::Metrics(_) => {}
            }
        }
        println!();
    } else {
        println!("\nAssistant: {assistant_content}");
    }

    Ok(())
}
