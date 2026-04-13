//! Weather MCP Server Example
//!
//! A simple MCP server that provides weather information.
//! Run with: cargo run --package arey-mcp --example mcp_weather --features server

use std::env;

use anyhow::Result;
use arey_mcp::mock::WeatherServer;
use rmcp::{service::ServiceExt, transport::stdio};

#[tokio::main]
async fn main() -> Result<()> {
    if env::args().len() < 1 || env::args().nth(1).unwrap() != "serve" {
        println!(
            "Usage: cargo run --package arey-mcp --example mcp_weather --features server -- serve"
        );
        return Ok(());
    }

    let service = WeatherServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
