# Design

Notes on the design of this tool.

## Goals

- Simplify development with local large language models. Must be usable both as
  a CLI app and a library.
- Task oriented.
- Must support CPU based ggml models.
- Must be extensible for following dimensions. Preferably with configuration.
  - Newer models
  - Prompt formats
- Must support integration with other tools with in the ecosystem.
- Opinionated. Explicitly avoid dependency or feature bloat.

### Non Goals

- `arey` is not a development library or SDK.
- No support for training or inference for generic models.

## Architecture

### Workspace Structure

This is a Rust monorepo with three main crates:

1. **`crates/core`** - Domain logic and foundational interfaces
   - Contains `agent`, `completion`, `config`, `model`, `session`, `tools` modules
   - Provides abstractions for LLM providers and capabilities
   - Core infrastructure for agents, workflows, and tools

2. **`crates/arey`** - CLI application binary
   - Main entry point with commands: `run`, `play`, `chat`
   - User experience components (styling, spinners) for output formatting
   - REPL engine with session management, context-aware autocomplete
   - REPL commands such as `/log`, `/tool`, `@agent`, `!workflow`

3. **`crates/tools-*`** - Tool implementations
   - E.g., `tools-search` for web search functionality
   - Each tool can be used inline or as independent MCP server

### Key Concepts

**Agent**: Stateless configuration defining persona and capabilities, configured in YAML files. Bundles system prompt, tool list, and model parameters.

**Session**: Stateful conversation object instantiated from an Agent, holding complete message history and tool state.

**Tools**: Extend agent functionality (search, memory, etc.). Dynamically managed per session without requiring model reload.

**Workflows**: Predefined sequences of agent/tool invocations for complex automation tasks.

**REPL Engine**: Interactive chat environment with:
- Context-aware autocomplete
- Command support (`/log`, `/tool`, `@agent`, `!workflow`)
- Session management and persistence

### Agent Configuration

Agents are defined in YAML files in the `~/.config/arey/agents/` directory. The application validates configuration on startup.

1. Startup: Parse config → Build Config → Populate Agent Repository
2. User Interaction: CLI → Agent Repository → Session Configuration → Stateful Session
3. Tools are sent with each completion request (not in system prompt)

Agents are defined declaratively in individual YAML files. This allows for easy creation and management of reusable agent personas.

**Example agent file (`~/.config/arey/agents/coder.yml`):**

```yaml
name: "coder"
prompt: "You are an expert Rust programmer. You only write concise, idiomatic Rust code."
tools:
  - search
profile:
  temperature: 0.3
  top_p: 0.9
```

**Example agent file (`~/.config/arey/agents/researcher.yml`):**

```yaml
name: "researcher"
prompt: "You are a meticulous researcher who always cites sources."
tools:
  - search
profile:
  temperature: 0.2
  top_p: 0.1
```

### Execution Flows

**Session Initialization and Interaction**

```mermaid
graph TD
    subgraph Startup
        ConfigFile(arey.yml) --> AppInit[Application Startup]
        AppInit -->|Parses configs| Config(Config Object)
        Config -->|Populates| AgentRepo[Agent Repository]
    end

    subgraph "User Interaction"
        User((User)) -->|"arey chat --agent coder"| CLI
        CLI -->|Gets "coder" from| AgentRepo
        AgentRepo -->|Returns| AgentConfig(AgentConfig)
        CLI -->|Instantiates session with config| Session(Session State)
        User <-->|Interact| Session
    end
```

### Design Decisions

To maintain simplicity and deliver core value incrementally, the following design decisions have been made:

- **Agent-to-Agent Invocation is Deferred**: To avoid the complexity of nested execution (context propagation, error handling, circular dependencies), the initial implementation will **not** support agents invoking other agents. This powerful feature is deferred for a future release, allowing the core user-to-agent interaction to be solidified first.

- **Dynamic Tool Management**: Tools are not part of the system prompt. Instead, the available toolset for a given task will be sent with each completion request to the LLM. This provides maximum flexibility for dynamically adding or removing tools during a session without needing to reload the model or manage complex state.

- **Clear Separation of Concerns**: The architecture maintains a strict separation between the stateless `Agent` configuration (the template) and the stateful `Session` (the conversation instance). This ensures that agent definitions are reusable and that session state is managed predictably.

- **Startup Configuration Validation**: To prevent runtime errors from misconfiguration, the application validates configuration files on startup. This includes checks to ensure that all tools referenced by an agent (e.g., `search`) correspond to actual, registered tool implementations. This surfaces errors to the user early.

- **Multi-Source Agent Loading**: Agents can be loaded from multiple sources with clear precedence rules:
  - **Built-in agents**: Embedded in the binary for common use cases
  - **User agents**: Individual YAML files in `~/.config/arey/agents/`
  - **Legacy agents**: From the `agents` section in `arey.yml` (deprecated)

  Precedence order: User agents > Built-in agents > Legacy agents. This allows users to override built-in agents while maintaining backward compatibility.

- **Performance Focus**: The implementation prioritizes CPU performance and low memory usage, making it suitable for running on modest hardware while maintaining responsive interactions with large language models.

### Example Runs

**1. Agent-Based Research**

```bash
arey run @research "latest AI advancements"
```

1. Agent searches academic/public sources
2. Summarizes key findings
3. Provides search references

**2. Documentation Workflow**

```bash
arey run !generate_docs src/
```

1. Index source files
2. Generate API documentation stubs
3. Verify coverage
4. Output markdown

**3. Interactive Troubleshooting**

```bash
arey chat
> @support "Connection timeout error"
```

1. Support agent suggests diagnostics
2. Integrates log analysis tools
3. Provides repair steps
4. Maintains session state
