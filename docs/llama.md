# Working with Llama models

This guide provides step-by-step guidance on using [Llama.cpp][] based models in
`arey`. We will download a model from Huggingface, configure `arey` to use the
model and finally run a few commands.

[Ollama][] is an alternate way to automatically download your models and run them
locally. See our [Ollama guide](ollama.md) if you prefer an automated quick
start.

[Llama.cpp]: https://github.com/ggerganov/llama.cpp
[Ollama]: https://ollama.com

## Concepts

Llama is a family of large language models (LLMs) provided by Meta. While this was the
initial open weights architecture available for local use, Mistral and their
fine-tunes are quite popular too.

LLMs are represented by the number of parameters they were trained with. E.g.,
7B, 13B, 33B and 70B are usual buckets. Size of the model increases with the
training parameters.

Llama.cpp provides a quantization mechanism (GGUF) to reduce the file size and
allows running these models on smaller form factors including CPU-only devices.
You will see quantization in the model name. E.g., `Q4_K_M` implies 4-bit
quantization.

!!! tip "Choosing a quantization"

    Always choose the _lower quantization_ of a _higher param_ model. E.g.,
    Q4_K_M of 13B is better than Q8_K_M of 7B.

## Get the models

Please use [Huggingface search](https://huggingface.co/models?search=gguf) to
find the GGUF models.

Ollama maintains a [registry](https://ollama.com/library) of their supported models.

<https://www.reddit.com/r/LocalLLaMA/> is a fantastic community to stay
updated and learn more about local models.

??? note "Our favorite models"

    | Model                      | Parameters | Quant  | Purpose      |
    |----------------------------|------------|--------|--------------|
    | [OpenHermes-2.5-Mistral][] | 7B         | Q4_K_M | General chat |
    | [Deepseek-Coder-6.7B][]    | 7B         | Q4_K_M | Coding       |
    | [NousHermes-2-Solar-10.7B] | 11B        | Q4_K_M | General chat |

[OpenHermes-2.5-Mistral]: https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
[Deepseek-Coder-6.7B]: https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF
[NousHermes-2-Solar-10.7B]: https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF

After you locate the huggingface repository, please download the model locally.
Here's an example to download the [Tiny Dolphin][] model.

[Tiny Dolphin]: https://huggingface.co/s3nh/TinyDolphin-2.8-1.1b-GGUF

```sh
$ mkdir ~/models
$ cd ~/models

# If wget is not available on your platform, open the below link
# in your browser and save it to ~/models.
# Size of this model: ~668MB
$ wget https://huggingface.co/s3nh/TinyDolphin-2.8-1.1b-GGUF/resolve/main/tinydolphin-2.8-1.1b.Q4_K_M.gguf

# ...
$ ls
tinydolphin-2.8-1.1b.Q4_K_M.gguf
```

## Configure

Let's add an entry for this model in arey's config file.

```yaml linenums="1" hl_lines="7-10 38 41"
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
  model: tinydolphin
  profile: precise
task:
  model: tinydolphin
  profile: precise
```

Noteworthy changes to the configuration file:

1. **Line 7-10**: we added a new model definition with the path of the downloaded model.
2. **Line 15**: we instruct `arey` to use `tinydolphin` model for chat.
3. **Line 18**: `arey` will use `tinydolphin` for the queries in ask command.

## Usage

You can use `chat`, `ask` or `play` commands to run this model.

### Completion settings

You can use **profiles** to configure `arey`. A profile is a collection of
settings used for tuning the AI model's response. Usually it includes following
parameters:

| Parameter      | Value   | Purpose                                      |
| -------------- | ------- | -------------------------------------------- |
| max_tokens     | 512     | Maximum number of tokens to generate         |
| repeat_penalty | 1-2     | Higher value discourages repetition of token |
| stop           | []      | Comma separated list of stop words           |
| temperature    | 0.0-1.0 | Lower temperature implies precise response   |
| top_k          | 0-30    | Number of tokens to consider for sampling    |
| top_p          | 0.0-1.0 | Lower value samples from most likely tokens  |

See the list of all parameters in [create_completion][] API documentation.

[create_completion]: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion

### Chatting with the model

Let's run `arey chat` to start a chat session. See below for an illustration.

```sh
❯ arey chat
Welcome to arey chat!
Type 'q' to exit.

✓ Model loaded. 0.13s.

How can I help you today?
> Who are you?

I am an artificial intelligence model that has been programmed to simulate human behavior, emotions, and responses based on data gathered from various sources. My primary goal is to provide
assistance in various areas of life, including communication, problem-solving, decision-making, and learning.


◼ Completed. 0.49s to first token. 2.10s total. 75.58 tokens/s. 159 tokens. 64 prompt tokens.
>
```

See [quickstart](index.md) for an example of `ask` and `play` commands.
