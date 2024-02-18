# Configure arey

This article is a detailed reference for the `arey` configuration file.

## Location

Arey stores the configuration file in following locations:

- `~/.config/arey/arey.yml` for Linux, MAC, or WSL2 in Windows.
- `C:\users\<username>\.arey\arey.yml` for Windows.

Configuration file is a valid YAML file.

A default configuration file is created by `arey` on the first run. You can
refer [here][config-file] for the latest configuration file snapshot.

[config-file]: https://github.com/codito/arey/blob/master/arey/data/config.yml

## Sections

### Models

Model section provides a list of local LLM models for [Llama.cpp][] or [Ollama][]
backends.

[Llama.cpp]: https://github.com/ggerganov/llama.cpp
[Ollama]: https://ollama.com

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
  ollama-tinydolphin:
    name: tinydolphin:latest # name of the model, see http://localhost:11434/api/tags
    type: ollama
    template: chatml
```

`models` is a list of following elements. The `key`, e.g., `tinydolphin` can be
user specified, it is used further to reference the model setting.

- `name` (only for Ollama): specify the ollama model name. Should be a valid
  name in the local library. Match with value from
  <http://localhost:11434/api/tags>.
- `path` (only for Llama.cpp): specify the model file path. Supports expansion
  of user directory marker `~`.
- `type`: must be `ollama` for Ollama models. Any other value is considered as
  `llama` model.
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

**Ollama models**: see the list of all parameters in [Model file][] API documentation.

**Llama.cpp models**: see the list of all parameters in [create_completion][] API documentation.

[Model file]: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
[create_completion]: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion

### Chat and Task settings

`chat` and `task` settings specify the model and profile for the `chat` and
`ask` commands respectively.

```yaml
chat:
  model: ollama-tinydolphin
  profile: precise
task:
  model: ollama-tinydolphin
  profile: precise
  settings:
    host: http://localhost:11434/
```

For either section, you can specify the model settings in a `settings` member.
Following model settings are supported for each model type.

**Ollama models**

| Setting key | Value                   | Remark                     |
| ----------- | ----------------------- | -------------------------- |
| host        | http://localhost:11434/ | Base url for Ollama server |

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
