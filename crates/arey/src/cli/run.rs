use anyhow::{Context, Result};
use arey_core::{
    completion::{Completion, CompletionMetrics},
    config::Config,
};
use futures::StreamExt;
use std::io::Write;

use crate::{
    cli::ux::{ChatMessageType, format_footer_metrics, style_chat_text},
    svc::run::Task,
};

/// Executes the run command with a given instruction and optional model override.
pub async fn execute(
    instruction: Vec<String>,
    model: Option<String>,
    config: &Config,
) -> Result<()> {
    let instruction = instruction.join(" ");
    let ask_model_config = if let Some(model_name) = model {
        config
            .models
            .get(model_name.as_str())
            .cloned()
            .context(format!("Model '{model_name}' not found in config."))?
    } else {
        config.task.model.clone()
    };
    let agent_name = config.task.agent_name.clone();
    let mut task = Task::new(instruction, ask_model_config, config, agent_name)?;

    eprintln!(
        "{}",
        style_chat_text("Loading model...", ChatMessageType::Footer)
    );
    let model_metrics = task.load_model().await?;
    eprintln!(
        "{}",
        style_chat_text(
            format!("Model loaded in {:.2}ms", model_metrics.init_latency_ms).as_str(),
            ChatMessageType::Footer
        )
    );

    eprintln!(
        "{}",
        style_chat_text("Generating response...", ChatMessageType::Footer)
    );
    eprintln!();

    let mut stream = task.run().await?;

    // Collect the response and metrics
    let mut metrics = CompletionMetrics::default();
    let mut finish_reason = None;

    while let Some(result) = stream.next().await {
        match result? {
            Completion::Response(r) => {
                if let Some(reason) = r.finish_reason {
                    finish_reason = Some(reason);
                }
                print!("{}", r.text);
                std::io::stdout().flush()?;
            }
            Completion::Metrics(m) => metrics = m,
        }
    }

    let footer = format_footer_metrics(&metrics, finish_reason.as_deref(), false);
    eprintln!();
    eprintln!();
    eprintln!("{}", style_chat_text(&footer, ChatMessageType::Footer));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::config::get_config;
    use serde_json::json;
    use tempfile::NamedTempFile;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    fn mock_event_stream_body() -> String {
        let events = [
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": "Hello"},
                    "index": 0,
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": " world!"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }),
        ];
        let mut body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn create_temp_config_file(server_uri: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let config_content = format!(
            r#"
models:
  test-model:
    name: test-model
    provider: openai
    base_url: "{server_uri}"
    api_key: "MOCK_OPENAI_API_KEY"
profiles: {{}}
chat:
    model: test-model
task:
  model: test-model
"#,
        );
        std::io::Write::write_all(&mut file, config_content.as_bytes()).unwrap();
        file
    }

    #[tokio::test]
    async fn test_execute_success() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200)
            .set_body_bytes(mock_event_stream_body().as_bytes())
            .insert_header("Content-Type", "text/event-stream");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .mount(&server)
            .await;

        let config_file = create_temp_config_file(&server.uri());
        let config = get_config(Some(config_file.path().to_path_buf()))?;
        let instruction = vec!["test".to_string(), "instruction".to_string()];

        execute(instruction, None, &config).await?;

        Ok(())
    }
}
