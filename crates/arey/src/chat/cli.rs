// Handles user interaction for chat
use crate::chat::service::Chat;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(multicall = true)]
pub struct CliCommand {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug, Hash, PartialEq, Eq)]
pub enum Command {
    /// Clear chat history
    Clear,
    /// Show detailed logs for the last assistant message
    Log,
    /// Set tools for the chat session. E.g. /tool search
    #[command(alias = "t")]
    Tool {
        /// Names of the tools to use
        names: Vec<String>,
    },
    /// Exit the chat session
    #[command(alias = "q", alias = "quit")]
    Exit,
}

impl Command {
    pub async fn execute(self, session: Arc<Mutex<Chat<'_>>>) -> Result<bool> {
        match self {
            Command::Clear => {
                session.lock().await.clear_messages().await;
                println!("Chat history cleared");
            }
            Command::Log => {
                let chat_guard = session.lock().await;
                match chat_guard.get_last_assistant_message().await {
                    Some(ctx) => {
                        println!(
                            "\n=== TOOL CALLS ===\n{:#?}\n====================",
                            ctx.tools
                        );
                    }
                    None => println!("No logs available"),
                }
            }
            Command::Tool { names } => {
                let chat_guard = session.lock().await;
                match chat_guard.set_tools(&names).await {
                    Ok(()) => {
                        if names.is_empty() {
                            println!("Tools cleared.");
                        } else {
                            println!("Tools set: {}", names.join(", "));
                        }
                    }
                    Err(e) => {
                        eprintln!("Error setting tools: {e}");
                    }
                }
            }
            Command::Exit => {
                println!("Bye!");
                return Ok(false);
            }
        }
        Ok(true)
    }
}
