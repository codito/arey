use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rmcp::{
    ServiceExt,
    model::CallToolRequestParams,
    service::{RoleClient, RunningService},
    transport::TokioChildProcess,
};
use serde_json::Value;
use tokio::process::Command;
use tracing::{debug, info};

use crate::config::McpServerConfig;
use arey_core::tools::{Tool, ToolError};

pub struct McpClient {
    name: String,
    service: Arc<RunningService<RoleClient, ()>>,
    tools: Vec<Arc<dyn Tool>>,
}

impl McpClient {
    pub async fn new(name: String, config: &McpServerConfig) -> Result<Self> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args);
        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let transport = TokioChildProcess::new(cmd)?;
        let service = Arc::new(().serve(transport).await?);

        Self::with_service(name, service).await
    }

    pub async fn with_service(
        name: String,
        service: Arc<RunningService<RoleClient, ()>>,
    ) -> Result<Self> {
        let tools = Self::list_tools(&service, &name, service.clone()).await?;

        info!("MCP server '{}' connected with {} tools", name, tools.len());

        Ok(Self {
            name,
            service,
            tools,
        })
    }

    async fn list_tools(
        service: &Arc<RunningService<RoleClient, ()>>,
        server_name: &str,
        service_arc: Arc<RunningService<RoleClient, ()>>,
    ) -> Result<Vec<Arc<dyn Tool>>> {
        let tool_defs = service.list_all_tools().await?;
        let mut tools = Vec::new();

        for tool_def in tool_defs {
            let raw_name = tool_def.name.to_string();
            let name = format!("{}_{}", server_name, raw_name);
            let description = tool_def.description.unwrap_or_default().to_string();
            let input_schema = serde_json::to_value(&*tool_def.input_schema)?;

            let tool = Arc::new(McpTool::with_service(
                name,
                description,
                input_schema,
                Arc::new(server_name.to_string()),
                service_arc.clone(),
            )) as Arc<dyn Tool>;
            tools.push(tool);
        }

        Ok(tools)
    }

    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn call_tool(&self, tool_name: &str, arguments: &Value) -> Result<Value> {
        // Strip the server prefix to get the actual tool name
        let actual_tool_name = if tool_name.starts_with(&format!("{}_", self.name)) {
            tool_name.strip_prefix(&format!("{}_", self.name)).unwrap()
        } else {
            tool_name
        };
        let args_map = match arguments {
            Value::Object(m) => Some(m.clone()),
            _ => {
                let mut map = serde_json::Map::new();
                map.insert("input".to_string(), arguments.clone());
                Some(map)
            }
        };

        let mut params = CallToolRequestParams::default();
        params.meta = None;
        params.name = std::borrow::Cow::Owned(actual_tool_name.to_string());
        params.arguments = args_map;
        params.task = None;
        debug!(
            "MCP {}: calling tool {} with arguments {:?}",
            self.name, tool_name, arguments
        );
        let result = self.service.call_tool(params).await?;

        let output = result.structured_content.unwrap_or_else(|| {
            if let Some(content) = result.content.into_iter().next() {
                if let Some(text) = content.as_text() {
                    serde_json::json!({ "text": text.text })
                } else {
                    serde_json::json!({ "error": "Unsupported content type" })
                }
            } else {
                serde_json::json!({ "error": "Empty response" })
            }
        });

        Ok(output)
    }
}

struct McpTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    server_name: Arc<String>,
    #[allow(dead_code)]
    service: Option<Arc<RunningService<RoleClient, ()>>>,
}

impl McpTool {
    #[allow(dead_code)]
    pub fn new(
        name: String,
        description: String,
        input_schema: serde_json::Value,
        server_name: Arc<String>,
    ) -> Self {
        Self {
            name,
            description,
            input_schema,
            server_name,
            service: None,
        }
    }

    pub fn with_service(
        name: String,
        description: String,
        input_schema: serde_json::Value,
        server_name: Arc<String>,
        service: Arc<RunningService<RoleClient, ()>>,
    ) -> Self {
        Self {
            name,
            description,
            input_schema,
            server_name,
            service: Some(service),
        }
    }

    async fn execute_internal(&self, arguments: &Value) -> Result<Value> {
        let service = self.service.as_ref().ok_or_else(|| {
            anyhow::anyhow!("MCP service not available - tool created without service connection")
        })?;

        let actual_tool_name = if self.name.starts_with(&format!("{}_", self.server_name)) {
            self.name
                .strip_prefix(&format!("{}_", self.server_name))
                .unwrap()
        } else {
            &self.name
        };

        let args_map = match arguments {
            Value::Object(m) => Some(m.clone()),
            _ => {
                let mut map = serde_json::Map::new();
                map.insert("location".to_string(), arguments.clone());
                Some(map)
            }
        };

        let mut params = CallToolRequestParams::default();
        params.meta = None;
        params.name = std::borrow::Cow::Owned(actual_tool_name.to_string());
        params.arguments = args_map;
        params.task = None;
        let result = service.call_tool(params).await?;

        let output = result.structured_content.unwrap_or_else(|| {
            if let Some(content) = result.content.into_iter().next() {
                if let Some(text) = content.as_text() {
                    serde_json::json!({ "text": text.text })
                } else {
                    serde_json::json!({ "error": "Unsupported content type" })
                }
            } else {
                serde_json::json!({ "error": "Empty response" })
            }
        });

        Ok(output)
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn parameters(&self) -> Value {
        self.input_schema.clone()
    }

    async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        match self.execute_internal(arguments).await {
            Ok(result) => Ok(result),
            Err(e) => Err(ToolError::ExecutionError(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::tools::Tool;
    use serde_json::json;
    use yare::parameterized;

    fn create_test_tool() -> McpTool {
        McpTool::new(
            "test_tool".to_string(),
            "A test tool for testing".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                },
                "required": ["input"]
            }),
            Arc::new("test_server".to_string()),
        )
    }

    #[test]
    fn test_mcp_tool_name() {
        let tool = create_test_tool();
        assert_eq!(tool.name(), "test_tool");
    }

    #[test]
    fn test_mcp_tool_description() {
        let tool = create_test_tool();
        assert_eq!(tool.description(), "A test tool for testing");
    }

    #[test]
    fn test_mcp_tool_parameters() {
        let tool = create_test_tool();
        let params = tool.parameters();

        assert_eq!(params["type"], "object");
        assert!(params["properties"].is_object());
        assert!(params["required"].is_array());
    }

    #[tokio::test]
    async fn test_mcp_tool_execute_returns_error() {
        let tool = create_test_tool();
        let arguments = json!({ "input": "hello" });

        let result = tool.execute(&arguments).await;

        // Without service connection, should return error
        assert!(result.is_err() || result.as_ref().ok().and_then(|v| v.get("error")).is_some());
    }

    #[tokio::test]
    async fn test_mcp_tool_execute_empty_args() {
        let tool = create_test_tool();
        let arguments = json!({});

        let result = tool.execute(&arguments).await;

        // Without service connection, should return error
        assert!(result.is_err() || result.as_ref().ok().and_then(|v| v.get("error")).is_some());
    }

    #[test]
    fn test_mcp_tool_different_inputs() {
        let tool = McpTool::new(
            "read_file".to_string(),
            "Read a file from the filesystem".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path" }
                }
            }),
            Arc::new("fs".to_string()),
        );

        assert_eq!(tool.name(), "read_file");
        assert_eq!(tool.description(), "Read a file from the filesystem");
        assert!(tool.parameters().is_object());
    }

    #[test]
    fn test_mcp_tool_impl_tool_trait() {
        // Verify McpTool implements the Tool trait
        let tool: Box<dyn Tool> = Box::new(create_test_tool());

        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool for testing");
        assert!(tool.parameters().is_object());
    }

    #[parameterized(
        strip_prefix = { "server1_echo", "server1", "echo" },
        fs_read = { "fs_read_file", "fs", "read_file" },
        memory_context = { "memory_get_context", "memory", "get_context" },
        no_prefix = { "echo", "server1", "echo" },
        exact_match = { "server1", "server1", "server1" },
        different_server = { "other_add", "server1", "other_add" },
    )]
    fn test_prefix_stripping_logic(tool_name: &str, server_name: &str, expected: &str) {
        let actual = if tool_name.starts_with(&format!("{}_", server_name)) {
            tool_name
                .strip_prefix(&format!("{}_", server_name))
                .unwrap()
        } else {
            tool_name
        };
        assert_eq!(actual, expected);
    }
}
