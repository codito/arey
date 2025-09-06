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

We're not going to be build yet another index, or a semantic kernel like
`langchain`. We focus only on the language modeling aspect; not the storage or
anything else.

While these are true at the time of writing, we will be open to reconsider these
in the future.

- Training models are not supported. This is an inference library.
- GPU support is not available (due to lack of resource/testing environment).

### Better tools

We believe below tools are awesome and may be in the same bucket as this one.

- <https://github.com/simonw/llm>
- <https://github.com/go-skynet/LocalAI>
- <https://github.com/jmorganca/ollama>

## Architecture

### Clean Architecture Approach

We implement a layered architecture with clear separation of concerns:

1. **Core Crate**: Contains domain abstractions and foundational interfaces:
   - `Agent`: Traits for agent definitions and execution
   - `Workflow`: Step-by-step automation engines
   - `Tool`: Standardized tool interface
   - `Memory`: Short/long-term contextual storage

2. **Implementation Crates**: Concrete implementations in dedicated crates:
   - `agent-research`: Domain-specific agent
   - `memory-persistent`: Persistent storage
   - `tool-search`: Web search implementation

3. **CLI Crate (`arey`)**: Provides user experience with:
   - Top-level commands: `run`, `play`, `chat`
   - REPL engine with commands: `/log`, `/tool`, `@agent`, `!workflow`
   - Managed sessions with state persistence
   - Consistent UX components (styling, spinners)

### Key Concepts

- **Agent**: A stateless configuration that defines a persona and capabilities, configured in `arey.yml`. It bundles a system prompt, a list of default tool names, and model generation parameters (`ProfileConfig`). Agents are reusable templates for creating specialized conversational experiences.

- **Session**: A stateful, long-lived object that represents a single, continuous conversation. A session is instantiated from an `Agent` configuration and holds the complete message history. It is the primary entity that a user interacts with.

- **Nested Execution (Future)**: The architecture is designed to support delegating sub-tasks to specialized agents in the future. In this model, a parent session would create a temporary, isolated child session for a sub-task. This capability is currently deferred to simplify the initial implementation.

- **Tools**: Extend agent functionality through integrations like web search or file operations. A session's toolset is initialized from its agent, but can be dynamically modified for the duration of the session.

- **Workflows**: Predefined sequences of agent and tool invocations to automate complex, multi-step tasks.

- **REPL Engine**: Interactive chat environment with:
  - Command history
  - Context-aware autocomplete
  - Rich output formatting for agent and tool responses.

### Agent Configuration and Discovery

Agents are defined declaratively in the `arey.yml` configuration file under a top-level `agents` map. This allows for easy creation and management of reusable agent personas.

**Discovery**: At startup, the application parses the `agents` section and populates an internal Agent Repository. When a user requests an agent (e.g., via the `--agent` flag), the application retrieves the corresponding configuration from this repository to initialize a session.

**Example `arey.yml`:**
```yaml
agents:
  coder:
    prompt: "You are an expert Rust programmer. You only write concise, idiomatic Rust code."
    tools:
      - search
    profile: concise

  researcher:
    prompt: "You are a meticulous researcher who always cites sources."
    tools:
      - search
    profile:
      # An inline profile can also be used
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

- **Startup Configuration Validation**: To prevent runtime errors from misconfiguration, the application will validate the `arey.yml` file on startup. This includes checks to ensure that all tools referenced by an agent (e.g., `search`) correspond to actual, registered tool implementations. This surfaces errors to the user early.

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
