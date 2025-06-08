mod core;
mod platform;

use crate::core::chat::Chat;
use crate::core::config::get_config;
use crate::core::model::ModelConfig;
use anyhow::Context;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = get_config(None).context("Failed to load configuration")?;

    // Start chat with the configured chat model
    let chat_model_config = config.chat.model;
    let chat_instance = Chat::new(chat_model_config).await?;
    start_chat(chat_instance).await?;

    Ok(())
}

async fn start_chat(mut chat: Chat) -> anyhow::Result<()> {
    println!("Welcome to arey chat! Type 'q' to exit.");

    loop {
        print!("> ");
        use std::io::{self, Write};
        io::stdout().flush()?; // Ensure prompt is displayed immediately

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input == "q" || user_input == "quit" {
            println!("Bye!");
            break;
        }

        let mut stream = chat.stream_response(user_input.to_string()).await?;

        while let Some(response_result) = stream.next().await {
            match response_result {
                Ok(chunk) => {
                    print!("{}", chunk.text);
                }
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        println!(); // Newline after AI response
    }

    Ok(())
}
