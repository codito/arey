# Configure arey

This article is a detailed reference for the `arey` configuration file.

## Location

Arey stores the configuration file in following locations:

- `~/.config/arey/arey.yml` for Linux, MAC, or WSL2 in Windows.
- `C:\users\<username>\.arey\arey.yml` for Windows.

Configuration file is a valid YAML file.

A default configuration file is created by `arey` on the first run. You can
refer [here][config-file] for the latest configuration file snapshot.

[config-file]: https://github.com/codito/arey/blob/master/crates/core/data/config.yml

## Sections

### Models

Model section provides a list of local LLM models for [Llama.cpp][] or [OpenAI][]
compatible backends.

[Llama.cpp]: https://github.com/ggerganov/llama.cpp

```yaml
# Example configuration for arey app
#
# Looked up from XDG_CONFIG_DIR (~/.config/arey/arey.yml) on Linux or
# ~/.arey on Windows.

models:
  tinydolphin:
    path: ~/models/tinydolphin-2.8-1.1b.Q4_K_M.gguf
    type: llama
    template: chatml
```

`models` is a list of following elements. The `key`, e.g., `tinydolphin` can be
user specified, it is used further to reference the model setting.

- `name` (only for OpenAI compatible service): specify the model name.
- `path` (only for Llama.cpp): specify the model file path. Supports expansion
  of user directory marker `~`.
- `type`: must be `openai` for OpenAI compatible API endpoints. Any other value
  is considered as `llama` model.
- `template`: conversation template used by the model. We use this for the `arey
chat` command to convert user and assistant messages. See the Templates
  section below for details.

### Profiles

Profiles section is a collection of settings used for generating LLM response.
You can define as many profiles as necessary. The `key` e.g., `precise` is used
in other sections to refer to the settings.

```yaml
profiles:
  # See https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
  precise:
    # factual responses and straightforward assistant answers
    temperature: 0.7
    repeat_penalty: 1.176
    top_k: 40
    top_p: 0.1
  creative:
    # chatting, storywriting, and interesting assistant answers
    temperature: 0.72
    repeat_penalty: 1.1
    top_k: 0
    top_p: 0.73
  sphinx:
    # varied storywriting (on 30B/65B) and unconventional chatting
    temperature: 1.99
    repeat_penalty: 1.15
    top_k: 30
    top_p: 0.18
```

Each `profile` can specify the completion settings specific to the model type.
Below are a few common settings.

| Parameter      | Value   | Purpose                                      |
| -------------- | ------- | -------------------------------------------- |
| repeat_penalty | 1-2     | Higher value discourages repetition of token |
| stop           | []      | Comma separated list of stop words           |
| temperature    | 0.0-1.0 | Lower temperature implies precise response   |
| top_k          | 0-30    | Number of tokens to consider for sampling    |
| top_p          | 0.0-1.0 | Lower value samples from most likely tokens  |

**Llama.cpp models**: see the list of all parameters in [create_completion][] API documentation.

[create_completion]: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion

### Agents

Agents are the core building blocks of `arey` that define AI personas with specific
prompts, tool access, and behavioral settings. The unified Agent system combines
configuration, metadata, and runtime state into a single, cohesive structure.

#### Agent Configuration

Agents can be defined in multiple locations with the following precedence order:

1. **User agents**: `~/.config/arey/agents/*.yml` (highest precedence)
2. **Built-in agents**: Embedded in the binary
3. **Legacy agents**: `agents:` section in `arey.yml` (lowest precedence)

##### Agent File Format

Create agent files in `~/.config/arey/agents/` with the following format:

```yaml
# ~/.config/arey/agents/coder.yml
name: "coder"
prompt: "You are an expert Rust programmer. You only write concise, idiomatic Rust code."
tools:
  - search
  - file
profile:
  temperature: 0.3
  repeat_penalty: 1.1
```

##### Agent Fields

- `name` (String): Unique identifier for the agent
- `prompt` (String): System prompt that defines the agent's persona and behavior
- `tools` (List<String]): Tools available to the agent (e.g., "search", "file")
- `profile` (Object): LLM generation parameters that override defaults

##### Built-in Agents

`arey` includes a built-in default agent:

```yaml
# Built-in default agent
name: "default"
prompt: "You are a helpful assistant."
tools: []
```

#### Agent Runtime State

Each agent instance maintains runtime state that can be modified during sessions:

- **Active state**: Agents can be activated/deactivated
- **Model overrides**: Session-specific model selection
- **Profile overrides**: Session-specific parameter adjustments
- **Tool overrides**: Session-specific tool availability

#### Agent Usage

Use agents with the `@` syntax in both chat and task modes:

```bash
# Chat with a specific agent
arey chat
> @coder Write a function to calculate fibonacci numbers

# Run a task with a specific agent
arey ask "@coder Explain Rust's ownership system"
```

#### Example Agent Configurations

##### Research Agent
```yaml
# ~/.config/arey/agents/researcher.yml
name: "researcher"
prompt: |
  You are a meticulous researcher who always cites sources. When providing information,
  you mention where the information comes from and verify important claims.
tools:
  - search
profile:
  temperature: 0.2
  top_p: 0.1
```

##### Code Review Agent
```yaml
# ~/.config/arey/agents/reviewer.yml
name: "reviewer"
prompt: |
  You are an experienced code reviewer. You focus on:
  1. Code correctness and safety
  2. Performance and efficiency
  3. Readability and maintainability
  4. Best practices and idioms
tools:
  - file
  - search
profile:
  temperature: 0.1
  repeat_penalty: 1.2
```

#### Agent Metadata and Tracking

The system tracks agent metadata including:

- **Source**: Built-in vs user-defined agents
- **File path**: Location of user agent files
- **Loading precedence**: Which agent takes priority when multiple exist

### Chat and Task settings

`chat` and `task` settings specify the model, profile, and default agent for the
`chat` and `ask` commands respectively.

```yaml
chat:
  model: tinydolphin
  profile: precise
  agent: default
task:
  model: tinydolphin
  profile: precise
  agent: default
```

For either section, you can specify the model settings in a `settings` member.
Following model settings are supported for each model type.

**Llama.cpp models**

| Setting key  | Value             | Remark                             |
| ------------ | ----------------- | ---------------------------------- |
| n_threads    | Half of CPU count | Number of threads to run           |
| n_ctx        | 4096              | Context window size                |
| n_batch      | 512               | Batch size                         |
| n_gpu_layers | 0                 | Number of layers to offload to GPU |
| use_mlock    | False             | Lock the model in main memory      |
| verbose      | False             | Show verbose logs                  |

## Prompt templates

A prompt template allows to specify tokens that are replaced during the runtime.

See <https://github.com/codito/arey/blob/master/arey/data/prompts/chatml.yml>
for an example.
