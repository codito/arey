//! Mock MCP tools for testing.
//!
//! This module provides mock tools that can be used for testing MCP integration
//! without spawning external MCP servers.

#[cfg(any(test, feature = "test_utils"))]
pub mod test_helpers {
    use std::sync::Arc;

    use anyhow::Result;
    use async_trait::async_trait;
    use rmcp::schemars;
    use rmcp::{
        handler::server::wrapper::{Json, Parameters},
        schemars::JsonSchema,
        tool, tool_router,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    use arey_core::tools::{Tool, ToolError};

    /// Mock tool for testing - echoes back input
    #[derive(Clone)]
    pub struct MockEchoTool {
        server_name: String,
    }

    impl MockEchoTool {
        pub fn new(server_name: &str) -> Self {
            Self {
                server_name: server_name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for MockEchoTool {
        fn name(&self) -> String {
            format!("{}_echo", self.server_name)
        }

        fn description(&self) -> String {
            "Echo back the input (mock tool)".to_string()
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                }
            })
        }

        async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
            let input = arguments
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("mock echo result");
            Ok(json!({ "echoed": input }))
        }
    }

    /// Mock tool for testing - adds two numbers
    #[derive(Clone)]
    pub struct MockAddTool {
        server_name: String,
    }

    impl MockAddTool {
        pub fn new(server_name: &str) -> Self {
            Self {
                server_name: server_name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for MockAddTool {
        fn name(&self) -> String {
            format!("{}_add", self.server_name)
        }

        fn description(&self) -> String {
            "Add two numbers (mock tool)".to_string()
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                },
                "required": ["a", "b"]
            })
        }

        async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
            let a = arguments.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
            let b = arguments.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({ "sum": a + b }))
        }
    }

    /// Mock tool for testing - returns current time
    #[derive(Clone)]
    pub struct MockTimeTool {
        server_name: String,
    }

    impl MockTimeTool {
        pub fn new(server_name: &str) -> Self {
            Self {
                server_name: server_name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for MockTimeTool {
        fn name(&self) -> String {
            format!("{}_get_time", self.server_name)
        }

        fn description(&self) -> String {
            "Get the current time (mock tool)".to_string()
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(&self, _arguments: &Value) -> Result<Value, ToolError> {
            Ok(json!({ "time": "2024-01-01T00:00:00Z" }))
        }
    }

    /// Create mock MCP tools for testing
    pub fn create_mock_tools(server_name: &str) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::new(MockEchoTool::new(server_name)) as Arc<dyn Tool>,
            Arc::new(MockAddTool::new(server_name)) as Arc<dyn Tool>,
            Arc::new(MockTimeTool::new(server_name)) as Arc<dyn Tool>,
        ]
    }

    /// Create a test configuration with MCP servers for testing
    pub fn create_test_mcp_config() -> serde_yaml::Value {
        serde_yaml::from_str(
            r#"
servers:
  test:
    command: echo
    args: ["test"]
    enabled: true
"#,
        )
        .unwrap()
    }

    // --- Weather Server for Testing ---
    #[derive(Clone)]
    pub struct WeatherServer;

    #[derive(Serialize, Deserialize, JsonSchema)]
    pub struct AddRequest {
        pub a: i32,
        pub b: i32,
    }

    #[derive(Serialize, Deserialize, JsonSchema)]
    pub struct GetWeatherRequest {
        pub location: String,
    }

    #[derive(Serialize, Deserialize, JsonSchema)]
    pub struct GetWeatherResponse {
        pub location: String,
        #[serde(rename = "temp_C")]
        pub temp_c: i32,
        #[serde(rename = "weatherDesc")]
        pub weather_desc: String,
        pub humidity: i32,
        #[serde(rename = "precipMM")]
        pub precip_mm: i32,
    }

    #[tool_router(server_handler)]
    impl WeatherServer {
        #[tool(description = "Add two numbers")]
        fn add(&self, Parameters(req): Parameters<AddRequest>) -> String {
            (req.a + req.b).to_string()
        }

        #[tool(description = "Gets the current weather for a given location")]
        fn get_weather(
            &self,
            Parameters(req): Parameters<GetWeatherRequest>,
        ) -> Json<GetWeatherResponse> {
            Json(GetWeatherResponse {
                location: req.location,
                temp_c: 28,
                weather_desc: "sunny".to_string(),
                humidity: 54,
                precip_mm: 0,
            })
        }
    }
}

#[cfg(any(test, feature = "test_utils"))]
pub use test_helpers::*;
