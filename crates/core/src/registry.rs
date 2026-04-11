use crate::tools::Tool;
use std::sync::Arc;

/// Global registry for all available tools in the application.
///
/// # Design Rationale
///
/// ToolRegistry serves as the single source of truth for all tools.
/// It can hold different tool types:
/// - **Direct tools**: Implemented in Rust, executed locally
/// - **MCP tools** (future): Remote tools via MCP protocol
///
/// The Session doesn't need to know HOW a tool executes - it just calls
/// `tool.execute()`. The Tool trait abstracts away local vs remote execution.
///
/// # Usage
///
/// ```rust
/// use arey_core::registry::ToolRegistry;
/// use arey_core::tools::Tool;
/// use async_trait::async_trait;
/// use serde_json::json;
/// use std::sync::Arc;
///
/// // A dummy tool for example
/// struct ExampleTool;
/// #[async_trait]
/// impl Tool for ExampleTool {
///     fn name(&self) -> String { "example".to_string() }
///     fn description(&self) -> String { "An example tool".to_string() }
///     fn parameters(&self) -> serde_json::Value { json!({"type": "object"}) }
///     async fn execute(&self, _args: &serde_json::Value) -> Result<serde_json::Value, arey_core::tools::ToolError> {
///         Ok(json!({"result": "ok"}))
///     }
/// }
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(ExampleTool))?;
///
/// // Get specific tools by name
/// let tools = registry.tools_for(&["example".to_string()]);
/// assert_eq!(tools.len(), 1);
///
/// // Or get all tool names
/// let all_names = registry.list();
/// assert_eq!(all_names, vec!["example"]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ToolRegistry {
    tools: std::collections::HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: std::collections::HashMap::new(),
        }
    }

    /// Register a tool in the registry.
    ///
    /// Returns error if a tool with the same name is already registered.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> anyhow::Result<()> {
        let name = tool.name();
        if self.tools.contains_key(&name) {
            anyhow::bail!("Tool '{}' already registered", name);
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// List all registered tool names.
    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get a subset of tools by their names.
    ///
    /// Returns tools for all names that exist in the registry.
    /// Silently ignores names that don't exist.
    pub fn tools_for(&self, names: &[String]) -> Vec<Arc<dyn Tool>> {
        names
            .iter()
            .filter_map(|name| self.tools.get(name).cloned())
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Tool, ToolError};
    use async_trait::async_trait;
    use serde_json::Value;
    use serde_json::json;

    #[derive(Clone)]
    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> String {
            "dummy".to_string()
        }

        fn description(&self) -> String {
            "A dummy tool".to_string()
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _args: &Value) -> Result<Value, ToolError> {
            Ok(json!({"status": "ok"}))
        }
    }

    #[derive(Clone)]
    struct OtherTool;

    #[async_trait]
    impl Tool for OtherTool {
        fn name(&self) -> String {
            "other".to_string()
        }

        fn description(&self) -> String {
            "Another tool".to_string()
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _args: &Value) -> Result<Value, ToolError> {
            Ok(json!({"status": "ok"}))
        }
    }
    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(DummyTool);

        let result = registry.register(tool);
        assert!(result.is_ok());
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn test_registry_register_duplicate() {
        let mut registry = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(DummyTool);

        registry.register(tool.clone()).unwrap();
        let result = registry.register(tool);

        assert!(result.is_err());
    }

    #[test]
    fn test_registry_get() {
        let mut registry = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(DummyTool);
        registry.register(tool).unwrap();

        let retrieved = registry.get("dummy");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "dummy");
    }

    #[test]
    fn test_registry_tools_for() {
        let mut registry = ToolRegistry::new();
        let tool1: Arc<dyn Tool> = Arc::new(DummyTool);
        let tool2: Arc<dyn Tool> = Arc::new(OtherTool);
        registry.register(tool1).unwrap();
        registry.register(tool2).unwrap();

        let tools = registry.tools_for(&["dummy".to_string()]);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "dummy");
    }

    #[test]
    fn test_registry_tools_for_multiple() {
        let mut registry = ToolRegistry::new();
        let tool1: Arc<dyn Tool> = Arc::new(DummyTool);
        let tool2: Arc<dyn Tool> = Arc::new(OtherTool);
        registry.register(tool1).unwrap();
        registry.register(tool2).unwrap();

        let tools = registry.tools_for(&["dummy".to_string(), "other".to_string()]);
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_registry_tools_for_ignores_nonexistent() {
        let mut registry = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(DummyTool);
        registry.register(tool).unwrap();

        let tools = registry.tools_for(&["dummy".to_string(), "nonexistent".to_string()]);
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let mut registry = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(DummyTool);
        registry.register(tool).unwrap();

        let result = registry.get("nonexistent");
        assert!(result.is_none());
    }
}
