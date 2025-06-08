mod core;
mod platform;

use crate::core::chat::Chat;
use crate::core::model::{ModelConfig, ModelProvider};
use futures::StreamExt;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Hello, arey!");

    // Example usage of the new chat module
    start_chat().await?;

    Ok(())
}

async fn start_chat() -> anyhow::Result<()> {
    // This is a placeholder config. In a real app, you'd load this from a config file.
    let mut settings: HashMap<String, serde_yaml::Value> = HashMap::new();
    settings.insert("base_url".to_string(), "http://localhost:8080/v1".into()); // Replace with your OpenAI compatible API base URL
    settings.insert("api_key".to_string(), "sk-12345".into()); // Replace with your actual API key or "env:YOUR_ENV_VAR"

    let config = ModelConfig {
        name: "gpt-3.5-turbo".to_string(), // Or "llama2", "mistral", etc. depending on your Ollama/Llama.cpp setup
        r#type: ModelProvider::Openai,     // Or ModelProvider::Ollama, ModelProvider::Gguf
        capabilities: vec![crate::core::model::ModelCapability::Chat],
        settings,
    };

    println!("\nAttempting to create chat with model: {}", config.name);
    let mut chat = Chat::new(config).await?;

    println!("Chat created! Type 'q' to exit.");

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

        println!("\nAI Response:");
        let mut stream = chat.stream_response(user_input.to_string()).await?;

        while let Some(response_result) = stream.next().await {
            match response_result {
                Ok(chunk) => print!("{}", chunk.text),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        println!(); // Newline after AI response
    }

    Ok(())
}
