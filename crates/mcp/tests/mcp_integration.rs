use std::sync::Arc;

use anyhow::Result;
use arey_mcp::McpClient;
use arey_mcp::mock::WeatherServer;
use rmcp::service::ServiceExt;
use serde_json::json;
use tokio::io::duplex;

#[tokio::test]
async fn test_mcp_weather_server_in_process() -> Result<()> {
    // 1. Setup in-process transport using duplex
    let (client_io, server_io) = duplex(1024);

    // 2. Start MCP weather server on one end of the duplex
    tokio::spawn(async move {
        if let Ok(service) = WeatherServer.serve(server_io).await {
            let _ = service.waiting().await;
        }
    });

    // 3. Connect MCP client on the other end of the duplex
    let client_service = Arc::new(().serve(client_io).await?);
    let mcp_client = McpClient::with_service("weather".to_string(), client_service).await?;
    let mcp_tools = mcp_client.tools();

    assert_eq!(mcp_tools.len(), 2, "Should find 2 tools");

    // 4. Execute tool via McpClient::call_tool
    let args = json!({ "location": "London" });
    let result = mcp_client.call_tool("weather_get_weather", &args).await?;
    assert_eq!(result["location"], "London");
    assert!(result.get("temp_C").is_some(), "Should have temp_C field");

    // 5. Execute tool directly via Tool trait
    if let Some(tool) = mcp_tools.iter().find(|t| t.name() == "weather_get_weather") {
        let args = json!({ "location": "Tokyo" });
        let result = tool.execute(&args).await?;
        assert_eq!(result["location"], "Tokyo");
    } else {
        panic!("Tool weather_get_weather not found in registry");
    }

    // 6. Test add tool
    let add_args = json!({ "a": 5, "b": 3 });
    let add_result = mcp_client.call_tool("weather_add", &add_args).await?;
    // The server returns a string which McpClient wraps in {"text": "8"}
    assert_eq!(add_result["text"], "8");

    Ok(())
}
