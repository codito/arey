# Working with OpenAI compatible servers

This guide provides step-by-step guidance on using an OpenAI chat compatible
server endpoint with `arey`.

For other ways to run a local LLM, you can try one of the following:

- [Llama.cpp][] for loading GGUF models downloaded locally. See our [Llama guide](llama.md) if you prefer more customizable way to
  manage your models.
- [Ollama][] for managing local models. Please refer to the [Ollama
  guide](ollama.md).

[OpenAI]: https://platform.openai.com/docs/api-reference/chat
[Llama.cpp]: https://github.com/ggerganov/llama.cpp
[Ollama]: https://ollama.com

## Running the server

If you've [Llama.cpp][] installed on your machine, you can try the
`llama-server` binary to start a server.

```sh
> ./llama-server -co -m ~/docs/models/gemma-2/gemma-2-2b-it-Q6_K_L.gguf -ngl 99 --threads 11 -c 2048
```

Change the parameters as necessary. In this example, `ngl` offloads all model
layers to run on the GPU, `--threads` uses 11 threads on the machine and we use
context size of `2048` with `-c` parameter.

## Configure

Let's add an entry for this model in arey's config file.

```yaml linenums="1" hl_lines="11-14 38 41"
# Example configuration for arey app
#
# Looked up from XDG_CONFIG_DIR (~/.config/arey/arey.yml) on Linux or
# ~/.arey on Windows.

models:
  tinydolphin:
    path: ~/models/tinydolphin-2.8-1.1b.Q4_K_M.gguf
    type: llama
    template: chatml
  llamacpp-server:
    name: gemma-2-2b-it # model name
    type: openai
    template: gemma2

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

chat:
  model: llamacpp-server
  profile: precise
  settings:
    base_url: http://localhost:8080/
task:
  model: llamacpp-server
  profile: precise
  settings:
    base_url: http://localhost:8080/
```

Noteworthy changes to the configuration file:

1. **Line 11-14**: we added a new model definition with openai type.
2. **Line 38**: we instruct `arey` to use `llamacpp-server` model for chat.
3. **Line 41**: `arey` will use `llamacpp-server` for the queries in ask command.

## Usage

You can use `chat`, `ask` or `play` commands to run this model.

### Completion settings

You can use **profiles** to configure `arey`. A profile is a collection of
settings used for tuning the AI model's response. Usually it includes following
parameters:

| Parameter   | Value   | Purpose                                    |
| ----------- | ------- | ------------------------------------------ |
| temperature | 0.0-1.0 | Lower temperature implies precise response |

Currently, only `temperature` is supported for `openai` models. Watch this space
for more support soon!

### Chatting with the model

Let's run `arey chat` to start a chat session. See below for an illustration.

```sh
❯ arey chat
Welcome to arey chat!
Type 'q' to exit.

✓ Model loaded. 0.0s.

How can I help you today?
> Who are you?

I am an artificial intelligence model that has been programmed to simulate human behavior, emotions, and responses based on data gathered from various sources. My primary goal is to provide
assistance in various areas of life, including communication, problem-solving, decision-making, and learning.


◼ Completed. 0.49s to first token. 2.10s total.
>
```

See [quickstart](index.md) for an example of `ask` and `play` commands.
