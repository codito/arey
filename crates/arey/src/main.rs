use anyhow::Result;
use arey::commands::run_app;
use arey::ux;

#[tokio::main]
async fn main() -> Result<()> {
    if let Err(e) = run_app().await {
        ux::present_error(e);
        std::process::exit(1);
    }
    Ok(())
}
