use crate::cli::chat::commands::{CliCommand, parse_command_line};
use crate::cli::chat::compl::Repl;
use crate::cli::chat::prompt::format_status_prompt;
use crate::cli::ux::{
    ChatMessageType, GenerationSpinner, TerminalRenderer, format_footer_metrics, style_chat_text,
};
use crate::svc::chat::Chat;
use anyhow::Result;
use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, SenderType,
};
use arey_core::config::get_history_file_path;
use arey_core::tools::{Tool, ToolCall, ToolResult};
use clap::{CommandFactory, Parser};
use futures::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::{CompletionType, Editor};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error};

/// Runs the interactive REPL for the chat session.
pub async fn run(chat: Arc<Mutex<Chat<'_>>>, renderer: &mut TerminalRenderer<'_>) -> Result<()> {
    println!("Welcome to arey chat! Ask anything. Use '/help' for usage, '/q' to exit.");

    let config = rustyline::Config::builder()
        .completion_type(CompletionType::List)
        .history_ignore_space(true) // Ignore lines starting with space
        .auto_add_history(true) // Add new entries to history
        .build();

    let command_names = CliCommand::command()
        .get_subcommands()
        .flat_map(|c| c.get_name_and_visible_aliases())
        .map(|s| format!("/{s}"))
        .collect::<Vec<_>>();
    let tool_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_tools
            .clone()
            .keys()
            .map(|s| s.to_string())
            .collect()
    };
    let model_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_model_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    };
    let profile_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_profile_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    };

    let agent_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_agent_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    };

    let mut rl = Editor::with_config(config)?;

    // Try to set up file history, fall back to in-memory if it fails
    let history_file_path = match get_history_file_path() {
        Ok(path) => Some(path),
        Err(e) => {
            error!(
                "Warning: Could not create history file: {}. Using in-memory history.",
                e
            );
            None
        }
    };

    if let Some(ref path) = history_file_path
        && let Err(e) = rl.load_history(path)
    {
        error!(
            "Warning: Could not load history file: {}. Starting with empty history.",
            e
        );
    }

    rl.set_helper(Some(Repl {
        command_names,
        tool_names,
        model_names,
        profile_names,
        agent_names,
    }));

    // Helper function to save history on exit
    let save_history_on_exit = |rl: &mut Editor<_, _>| -> Result<()> {
        if let Some(ref path) = history_file_path
            && let Err(e) = rl.save_history(path)
        {
            error!("Warning: Could not save history file: {}", e);
        }
        Ok(())
    };

    loop {
        let prompt = {
            let chat_guard = chat.lock().await;
            format_status_prompt(&chat_guard)
        };
        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let trimmed_line = line.trim();

                if trimmed_line.is_empty() {
                    continue;
                }

                let is_command = trimmed_line.starts_with('/')
                    || trimmed_line.starts_with('!')
                    || trimmed_line.starts_with('@');

                if is_command {
                    let processed_args = parse_command_line(trimmed_line);

                    match CliCommand::try_parse_from(processed_args) {
                        Ok(cli_command) => {
                            if !cli_command.command.execute(chat.clone()).await? {
                                save_history_on_exit(&mut rl)?;
                                return Ok(()); // Exit REPL
                            }
                        }
                        Err(e) => {
                            // Use Clap's built-in help system
                            if e.kind() == clap::error::ErrorKind::DisplayHelp
                                || e.kind() == clap::error::ErrorKind::DisplayVersion
                            {
                                e.print()?;
                            } else {
                                // For other errors, still print them but suggest using --help
                                e.print()?;
                                eprintln!("Use '/command --help' for usage information.");
                            }
                        }
                    }
                } else {
                    let user_messages = vec![ChatMessage {
                        text: line.to_string(),
                        sender: SenderType::User,
                        ..Default::default()
                    }];
                    if !process_message(chat.clone(), renderer, user_messages).await? {
                        save_history_on_exit(&mut rl)?;
                        return Ok(());
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("Type /quit to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => {
                save_history_on_exit(&mut rl)?;
                println!("\nBye!");
                return Ok(());
            }
            Err(err) => {
                return Err(err.into());
            }
        }
    }
}

/// Recursively removes control characters from JSON values
fn clean_value(value: &mut serde_json::Value) {
    match value {
        Value::String(s) => {
            let cleaned = s.replace(|c: char| c.is_control(), "");
            *s = cleaned;
        }
        Value::Array(a) => a.iter_mut().for_each(clean_value),
        Value::Object(m) => m.values_mut().for_each(clean_value),
        _ => {}
    }
}

/// Process a message and generate a response.
async fn process_message(
    chat: Arc<Mutex<Chat<'_>>>,
    renderer: &mut TerminalRenderer<'_>,
    messages: Vec<ChatMessage>, // User input or tool responses
) -> Result<bool> {
    let mut metrics = CompletionMetrics::default();
    let mut finish_reason: Option<String> = None;

    // Clear renderer state for this new message processing cycle.
    renderer.clear();

    // Create spinner
    let spinner = GenerationSpinner::new("Generating...".to_string());
    let cancel_token = CancellationToken::new();

    // Clone for async block
    let chat_clone = chat.clone();

    // Store tool call responses if LLM requires a set of tools to be invoked for responding to a
    // user message.
    let mut assistant_message_text = String::new();
    let mut assistant_message_tools: Vec<ToolCall> = vec![];
    let mut assistant_tool_responses: Vec<ChatMessage> = vec![];

    let mut stream_error = false;
    let was_cancelled = {
        // Get stream response
        let mut chat_guard = chat_clone.lock().await;
        let available_tools = chat_guard.available_tools.clone();
        let mut stream = {
            chat_guard.add_messages(messages).await;
            chat_guard.stream_response(cancel_token.clone()).await?
        };

        let mut first_token_received = false;
        let mut was_cancelled_internal = false;

        // Start listening for Ctrl-C
        let mut ctrl_c_stream = Box::pin(tokio::signal::ctrl_c());

        // Process stream with Ctrl-C and tokenization detection
        loop {
            tokio::select! {
                // Ctrl-C handling
                _ = &mut ctrl_c_stream => {
                    cancel_token.cancel();
                    was_cancelled_internal = true;
                    break;
                },

                // Process the next stream token
                next = stream.next() => {
                    match next {
                        Some(response) => {
                            if !first_token_received {
                                spinner.clear();
                                first_token_received = true;
                            }

                            if cancel_token.is_cancelled() {
                                was_cancelled_internal = true;
                                break;
                            }

                            match response {
                                Ok(Completion::Response(chunk)) => {
                                    if !&chunk.text.is_empty() {
                                        assistant_message_text.push_str(&chunk.text);
                                        renderer.render_markdown(&chunk.text)?;
                                    }

                                    if let Some(reason) = &chunk.finish_reason {
                                        finish_reason = Some(reason.clone());
                                    }

                                    // Tool messages can come in chunks, we collate all
                                    if let Some(tools) = &chunk.tool_calls {
                                        assistant_message_tools.extend(tools.clone());
                                        assistant_tool_responses = process_tools(&available_tools, tools).await?;
                                    }
                                }
                                Ok(Completion::Metrics(m)) => {
                                    metrics = m;
                                }
                                Err(e) => {
                                    eprintln!("Error: {e}");
                                    stream_error = true;
                                    break;
                                }
                            }
                        }
                        // End of stream
                        None => break,
                    }
                }
            }
        }

        was_cancelled_internal || cancel_token.is_cancelled()
    };

    // Ensure spinner is cleared after stream processing
    spinner.clear();

    // If the stream produced an error, we're done. The error has already been printed.
    if stream_error {
        return Ok(true);
    }

    {
        // Add the assistant message to the chat history
        let mut chat_guard = chat.lock().await;
        chat_guard
            .add_messages(vec![ChatMessage {
                sender: SenderType::Assistant,
                text: assistant_message_text,
                tools: Some(assistant_message_tools),
                metrics: Some(metrics.clone()),
            }])
            .await;
    }

    // After a successful stream, flush any remaining partial lines from the renderer.
    renderer.render_markdown("\n")?;

    // If the model produced tool calls, recursively call this function to process them.
    if !assistant_tool_responses.is_empty() {
        return Box::pin(process_message(
            chat_clone,
            renderer,
            assistant_tool_responses,
        ))
        .await;
    }

    // If we've reached this point, the response is complete. Print the footer.
    let (metrics, finish_reason_option) = match was_cancelled {
        true => (CompletionMetrics::default(), None),
        false => (metrics, finish_reason),
    };

    let footer = format_footer_metrics(&metrics, finish_reason_option.as_deref(), was_cancelled);

    // The `render_markdown("\n")` above ensures we start on a fresh line.
    println!();
    println!("{}", style_chat_text(&footer, ChatMessageType::Footer));

    Ok(true)
}

/// Returns set of tool results as messages
async fn process_tools(
    available_tools: &HashMap<&str, Arc<dyn Tool>>,
    tool_calls: &Vec<ToolCall>,
) -> Result<Vec<ChatMessage>> {
    let mut tool_messages: Vec<ChatMessage> = vec![];

    for call in tool_calls {
        eprintln!(); // Add a newline before tool output
        let tool_fmt = format!("Tool: {}({})", call.name, call.arguments);
        let tool_msg = style_chat_text(&tool_fmt, ChatMessageType::Footer);
        let spinner = GenerationSpinner::new(tool_msg.to_string());
        let tool = match available_tools.get(call.name.as_str()) {
            Some(t) => t.clone(),
            None => {
                eprintln!(
                    "{}",
                    style_chat_text(
                        &format!("Tool '{}' not available", call.name),
                        ChatMessageType::Error
                    )
                );
                continue;
            }
        };

        // Normalize tool call arguments - could be direct JSON, escaped JSON, or plain string
        debug!("Raw tool call arguments: '{}'", call.arguments);

        let args = match serde_json::from_str(&call.arguments) {
            Ok(value) => {
                debug!("Parsed as direct JSON: {}", value);
                value
            }
            Err(first_error) => {
                debug!(
                    "First parse failed: {}. Trying raw string parse",
                    first_error
                );
                match serde_json::from_str::<Value>(&call.arguments) {
                    Ok(value) => {
                        debug!("Parsed as raw JSON string: {}", value);
                        value
                    }
                    Err(second_error) => {
                        debug!(
                            "Second parse failed: {}. Falling back to input wrapper",
                            second_error
                        );
                        serde_json::json!({ "input": call.arguments })
                    }
                }
            }
        };

        // If we got a string, try to parse that string as JSON to see if it's really a structured value.
        let args = match &args {
            Value::String(s) => match serde_json::from_str(s) {
                Ok(parsed_value) => {
                    debug!("Unescaped inner JSON string successfully: {}", parsed_value);
                    parsed_value
                }
                Err(inner_error) => {
                    debug!(
                        "Failed to unescape inner JSON: {}. Keeping as string.",
                        inner_error
                    );
                    args
                }
            },
            _ => args,
        };

        debug!("Final tool arguments: {}", args);
        let mut output = match tool.execute(&args).await {
            Ok(out) => out,
            Err(e) => {
                eprintln!(
                    "{}",
                    style_chat_text(
                        &format!("Tool execution failed: {e}"),
                        ChatMessageType::Error
                    )
                );
                continue;
            }
        };

        // Clean control characters from tool output
        clean_value(&mut output);

        spinner.clear();
        eprintln!("✓ {tool_msg}");

        // Gemini doesn't provide a tool_id, we fill it if empty
        let call_id = if call.id.is_empty() {
            call.name.to_string()
        } else {
            call.id.to_string()
        };
        let result = ToolResult {
            call: ToolCall {
                id: call_id,
                ..call.clone()
            },
            output,
        };
        tool_messages.push(ChatMessage {
            text: serde_json::to_string(&result)?,
            sender: SenderType::Tool,
            ..Default::default()
        });
    }

    eprintln!();
    Ok(tool_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ux::{TerminalRenderer, get_theme};
    use crate::svc::chat::Chat;
    use anyhow::Result;
    use arey_core::{
        completion::{ChatMessage, SenderType},
        tools::{Tool, ToolError, ToolResult},
    };
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::vec;
    use tokio::sync::Mutex;

    use crate::cli::chat::test_utils::{
        MockTool, create_test_config_with_custom_agent, create_test_config_with_error_model,
        create_test_config_with_tool_call_model,
    };

    #[tokio::test]
    async fn test_process_tools() -> Result<()> {
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool {});
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool.clone())]);
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "mock_tool".to_string(),
            arguments: "{}".to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;

        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.sender, SenderType::Tool);

        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;
        assert_eq!(tool_result.call.id, "call_1");
        assert_eq!(tool_result.output, json!("mock tool output"));

        Ok(())
    }

    #[tokio::test]
    async fn test_process_tools_with_stringified_json_argument() -> Result<()> {
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

        let tool: Arc<dyn Tool> = Arc::new(ArgRecorder);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("arg_recorder", tool.clone())]);
        // The arguments are a string that itself is a JSON object
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "arg_recorder".to_string(),
            arguments: r#""{\"arg\":42}""#.to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;

        // The tool should have received the parsed JSON object: {"arg":42}
        assert_eq!(tool_result.output, json!({"arg":42}));

        Ok(())
    }

    #[tokio::test]
    async fn test_process_tools_with_non_json_string() -> Result<()> {
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

        let tool: Arc<dyn Tool> = Arc::new(ArgRecorder);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("arg_recorder", tool.clone())]);
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "arg_recorder".to_string(),
            arguments: "plain string".to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;

        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;

        // We expect the tool to have received: {"input": "plain string"}
        assert_eq!(tool_result.output, json!({ "input": "plain string" }));

        Ok(())
    }

    #[tokio::test]
    async fn test_process_tools_cleans_control_characters() -> Result<()> {
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

        let tool: Arc<dyn Tool> = Arc::new(ControlCharTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("control_tool", tool.clone())]);
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "control_tool".to_string(),
            arguments: "{}".to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;

        // Check that control characters were removed
        let output_str = tool_result.output.get("key").unwrap().as_str().unwrap();
        assert_eq!(output_str, "value with  control  characters");
        Ok(())
    }

    #[tokio::test]
    async fn test_process_message_simple_response() -> Result<()> {
        // 1. Setup Chat and Renderer
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));
        let mut buffer = Vec::new();
        let theme = get_theme("ansi");
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        // 2. Call process_message
        let user_message = ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            ..Default::default()
        };
        process_message(chat_session.clone(), &mut renderer, vec![user_message]).await?;

        // 3. Assert rendered output
        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("Hello world"));
        assert_eq!(2, chat_session.lock().await.get_all_messages().len());
        Ok(())
    }

    #[tokio::test]
    async fn test_process_message_with_tool_call() -> Result<()> {
        let config = create_test_config_with_tool_call_model()?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool)]);
        let chat = Chat::new(
            &config,
            Some("tool-call-model".to_string()),
            available_tools,
        )
        .await?;
        let chat_session = Arc::new(Mutex::new(chat));

        let mut buffer = Vec::new();
        let theme = get_theme("ansi");
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        let user_message = ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            ..Default::default()
        };
        process_message(chat_session.clone(), &mut renderer, vec![user_message]).await?;

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("Tool output is mock tool output"));
        assert_eq!(4, chat_session.lock().await.get_all_messages().len());
        Ok(())
    }

    #[tokio::test]
    async fn test_process_message_stream_error() -> Result<()> {
        let config = create_test_config_with_error_model()?;
        let chat = Chat::new(&config, Some("error-model".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));
        let mut buffer = Vec::new();
        let theme = get_theme("ansi");
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        let user_message = ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            ..Default::default()
        };
        process_message(chat_session.clone(), &mut renderer, vec![user_message]).await?;

        // Expect no output to renderer, error is printed to stderr
        let output = String::from_utf8(buffer).unwrap();
        assert!(
            output.is_empty(),
            "Output should be empty. Output: {}",
            output
        );
        assert_eq!(1, chat_session.lock().await.get_all_messages().len());
        Ok(())
    }

    #[test]
    fn test_history_file_path_from_config() {
        // This test verifies that the function from config.rs can be called
        // The actual functionality is tested in config.rs
        let result = arey_core::config::get_history_file_path();

        // The function may fail if environment variables are not set
        // which is expected in some test environments
        if let Ok(path) = result {
            assert!(path.ends_with("arey/history.txt"));
        }
    }
}
