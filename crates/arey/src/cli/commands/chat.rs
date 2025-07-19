use crate::chat::{Chat, run};
use crate::ux::{TerminalRenderer, get_theme};
use ::console::Term;
use anyhow::{Context, Result};
use arey_core::config::Config;
use arey_core::tools::Tool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub async fn execute<'a>(
    model: Option<String>,
    config: &'a Config,
    available_tools: HashMap<&'a str, Arc<dyn Tool>>,
) -> Result<()> {
    let chat = Chat::new(config, model, available_tools)
        .await
        .context("Failed to initialize chat!")?;
    let mut term = Term::stdout();
    let theme = get_theme(&config.theme);
    let mut renderer = TerminalRenderer::new(&mut term, &theme);

    run(Arc::new(Mutex::new(chat)), &mut renderer).await
}
