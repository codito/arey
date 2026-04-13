//! MCP Client Example
//!
//! A simple client that connects to an MCP weather server and calls its tools.
//! Run with: cargo run --package arey-mcp --example mcp_client

use std::collections::HashMap;

use anyhow::Result;
use arey_mcp::{McpClient, McpServerConfig};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== MCP Weather Client ===\n");

    // 1. Configure the MCP server connection (using the weather example server)
    let mcp_config = McpServerConfig {
        command: "cargo".to_string(),
        args: vec![
            "run".to_string(),
            "--package".to_string(),
            "arey-mcp".to_string(),
            "--example".to_string(),
            "mcp_weather".to_string(),
            "--features".to_string(),
            "server".to_string(),
            "--quiet".to_string(),
            "--".to_string(),
            "serve".to_string(),
        ],
        env: HashMap::new(),
        enabled: true,
    };

    // 2. Connect to the server
    println!("Connecting to weather server...");
    let mcp_client = McpClient::new("weather".to_string(), &mcp_config).await?;

    // 3. List available tools
    let mcp_tools = mcp_client.tools();
    println!("Found {} tools:", mcp_tools.len());
    for tool in &mcp_tools {
        println!(" - {}: {}", tool.name(), tool.description());
    }
    println!();

    // 4. Call a tool using McpClient
    println!("Calling 'weather_get_weather' tool...");
    let args = json!({ "location": "London" });
    let result = mcp_client.call_tool("weather_get_weather", &args).await?;

    println!("Response: {}", serde_json::to_string_pretty(&result)?);
    println!("\n✅ Client example finished!");

    Ok(())
}
