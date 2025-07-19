use anyhow::Result;
use arey_core::config::Config;
use arey_core::tools::Tool;
use std::collections::HashMap;
use std::sync::Arc;

pub async fn execute(
    model: Option<String>,
    config: &Config,
    available_tools: HashMap<&str, Arc<dyn Tool>>,
) -> Result<()> {
    crate::cli::chat::execute(model, config, available_tools).await
}
