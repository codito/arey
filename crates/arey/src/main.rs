use anyhow::Result;
use arey::cli::{run, ux};

#[tokio::main]
async fn main() -> Result<()> {
    if let Err(e) = run().await {
        ux::present_error(e);
        std::process::exit(1);
    }
    Ok(())
}
