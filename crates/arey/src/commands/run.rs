use std::collections::HashMap;

use anyhow::{Context, Result};
use arey_core::completion::{CancellationToken, Completion};
use arey_core::config::Config;
use arey_core::model::ModelConfig;
use arey_core::session::Session;
use futures::StreamExt;

pub async fn run_ask(instruction: &str, model_config: ModelConfig) -> Result<()> {
    // TODO: use profile settings
    let mut session = Session::new(model_config, "").await?;
    session.add_message(arey_core::completion::SenderType::User, instruction);
    let mut stream = session
        .generate(HashMap::new(), CancellationToken::new())
        .await?;
    while let Some(completion) = stream.next().await {
        match completion {
            Ok(Completion::Response(response)) => print!("{}", response.text),
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
    println!();
    Ok(())
}

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
    run_ask(&instruction, ask_model_config).await
}
