use crate::cli::chat::commands::{CliCommand, parse_command_line};
use crate::cli::chat::compl::Repl;
use crate::cli::chat::prompt::format_status_prompt;
use crate::cli::ux::{
    ChatMessageType, GenerationSpinner, TerminalRenderer, format_footer_metrics, style_chat_text,
};
use crate::svc::chat::Chat;
use anyhow::{Context, Result};
use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, SenderType,
};
use arey_core::config::get_history_file_path;
use arey_core::session::SessionEvent;
use clap::{CommandFactory, Parser};
use futures::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::{CompletionType, Editor};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::error;

/// Runs the interactive REPL for the chat session.
pub async fn run(chat: Arc<Mutex<Chat<'_>>>, renderer: &mut TerminalRenderer<'_>) -> Result<()> {
    println!("Welcome to arey chat! Ask anything. Use '/help' for usage, '/q' to exit.");

    // Load the session with a loading indicator
    {
        let chat_guard = chat.clone();
        let mut chat = chat_guard.lock().await;
        let model_name = chat.model_key();
        let spinner = GenerationSpinner::new(format!("Loading model '{}'...", model_name));
        chat.load_session()
            .await
            .context("Failed to load session")?;
        spinner.clear();
    }

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
            .available_tool_names()
            .into_iter()
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
    let mut spinner = GenerationSpinner::new("Generating...".to_string());
    let cancel_token = CancellationToken::new();

    // Clone for async block
    let chat_clone = chat.clone();

    let mut stream_error = false;
    let was_cancelled = {
        // Get stream response
        let mut chat_guard = chat_clone.lock().await;
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
                            if cancel_token.is_cancelled() {
                                was_cancelled_internal = true;
                                break;
                            }

                            match response {
                                Ok(SessionEvent::Token(Completion::Response(chunk))) => {
                                    if !first_token_received {
                                        spinner.clear();
                                        first_token_received = true;
                                    }

                                    if let Some(thought) = &chunk.thought {
                                        renderer.render_markdown(thought)?;
                                    }

                                    if !&chunk.text.is_empty() {
                                        renderer.render_markdown(&chunk.text)?;
                                    }

                                    if let Some(reason) = &chunk.finish_reason {
                                        finish_reason = Some(reason.clone());
                                    }
                                }
                                Ok(SessionEvent::Token(Completion::Metrics(m))) => {
                                    metrics = m;
                                }
                                Ok(SessionEvent::ToolStart { calls }) => {
                                    spinner.clear();
                                    let tool_info: Vec<_> = calls.iter().map(|c| format!("{}({})", c.name, c.arguments)).collect();
                                    let tool_names_str = format!("Executing tools: {}", tool_info.join(", "));
                                    let tool_msg = style_chat_text(&tool_names_str, ChatMessageType::Footer);
                                    spinner = GenerationSpinner::new(tool_msg.to_string());
                                }
                                Ok(SessionEvent::ToolEnd { results }) => {
                                    spinner.clear();

                                    for (call, succeeded) in &results {
                                        if *succeeded {
                                            eprintln!("{}", style_chat_text(&format!("✔  {}({})", call.name, call.arguments), ChatMessageType::Footer));
                                        } else {
                                            eprintln!("{}", style_chat_text(&format!("✘  {}({})", call.name, call.arguments), ChatMessageType::Footer));
                                        }
                                    }
                                    eprintln!();

                                    // Reset spinner for next potential generation turn
                                    spinner = GenerationSpinner::new("Generating...".to_string());
                                    first_token_received = false;
                                }
                                Ok(SessionEvent::CompactionStart) => {
                                    spinner.clear();
                                    spinner = GenerationSpinner::new("Compacting context...".to_string());
                                }
                                Ok(SessionEvent::CompactionEnd { .. }) => {
                                    spinner.clear();
                                    spinner = GenerationSpinner::new("Generating...".to_string());
                                    first_token_received = false;
                                }
                                Ok(_) => {} // Handle other events silently
                                Err(e) => {
                                    eprintln!("{}", style_chat_text(&format!("Error: {e}"), ChatMessageType::Error));
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

    // After a successful stream, flush any remaining partial lines from the renderer.
    renderer.render_markdown("\n")?;

    // If we've reached this point, the response is complete. Print the footer.
    let (metrics, finish_reason_option) = match was_cancelled {
        true => (CompletionMetrics::default(), None),
        false => (metrics, finish_reason),
    };

    let footer = format_footer_metrics(&metrics, finish_reason_option.as_deref(), was_cancelled);

    // The `render_markdown("\n")` above ensures we start on a fresh line.
    eprintln!();
    eprintln!("{}", style_chat_text(&footer, ChatMessageType::Footer));

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ux::{TerminalRenderer, get_theme};
    use crate::svc::chat::Chat;
    use anyhow::Result;
    use arey_core::{
        completion::{ChatMessage, SenderType},
        registry::ToolRegistry,
        tools::Tool,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;

    use crate::cli::chat::test_utils::{
        MockTool, create_test_config_with_custom_agent, create_test_config_with_error_model,
        create_test_config_with_tool_call_model,
    };

    #[tokio::test]
    async fn test_process_message_simple_response() -> Result<()> {
        // 1. Setup Chat and Renderer
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(
            &config,
            Some("test-model-1".to_string()),
            ToolRegistry::new(),
        )?;
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
        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(mock_tool)?;
        let mut chat = Chat::new(&config, Some("tool-call-model".to_string()), tool_registry)?;
        chat.load_session().await?;
        chat.set_tools(&["mock_tool".to_string()]).await?;
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
        let chat = Chat::new(
            &config,
            Some("error-model".to_string()),
            ToolRegistry::new(),
        )?;
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
