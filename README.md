# arey

Arey is a simple large language model app.

✓ Chat with your favorite local models. CPU friendly 🍀  
✓ Ask the AI model with a single command.  
✓ Supercharged in-context learning workflow ❤️ Edit your prompt in _any_ editor
and `arey` will generate a completion on save.  
✓ No telemetry, no internet, nothing to sell. Dedicated to the public domain.

🚧 Much more to come... See **Roadmap** below.

## Installation

```sh
# Install pipx if needed: `pip install pipx`
# Ensure ~/.local/bin is available in system PATH
pipx install arey
```

Please use [WSL][] for Windows installation. Troubleshooting notes are [here](docs/windows.md).

[WSL]: https://learn.microsoft.com/en-us/windows/wsl/install

## Usage

```sh
❯ arey --help
Usage: arey [OPTIONS] COMMAND [ARGS]...

  Arey - a simple large language model app.

Options:
  -v, --verbose BOOLEAN  Show verbose logs.
  --help                 Show this message and exit.

Commands:
  ask   Run an instruction and generate response.
  chat  Chat with an AI model.
  play  Watch FILE for model, prompt and generate response on edit.
```

On the first run, `arey` will create a configuration file in following location:

- `~/.config/arey/arey.yml` for Linux or Mac systems.
- `~/.arey/arey.yml` for Windows.

Please update the `models` section in the config yml to your local model path.

**Ask Arey something!**

`arey ask "Who is Seneca? Tell me one of his teachings"`

**Chat with Arey** using `arey chat`.

**Run Arey in play mode** to fine-tune a prompt in your editor while `arey` keeps
completing your prompt on every save.

```sh
❯ arey play /tmp/arey_playzl9igj3d.md

Welcome to arey play! Edit the play file below in your favorite editor and I'll generate a
response for you. Use `Ctrl+C` to abort play session.

Watching `/tmp/arey_playzl9igj3d.md` for changes...

───────────────────────────────────── 2024-01-21 17:20:01 ──────────────────────────────────────
✓ Model loaded. 0.57s.

Life is short because it passes by quickly and can end at any moment. We should make the most of
our time here on earth and live a virtuous life according to stoicism.

◼ Canceled.

Watching `/tmp/arey_playzl9igj3d.md` for changes...
```

## Development

```sh
# Install myl locally in editable mode.
> pip install -e .
> pip install -e .\[test\] # optional, if you wish to run tests

# Install with samples dependency if you wish to run them
> pip install -e .\[samples\]
```

With OPENBLAS, loading time for models is much smaller and inference is about
4-5% faster. Here's how to install `llama-cpp-python` with OPENBLAS support:

```sh
> pip uninstall llama-cpp-python
> CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --verbose
```

If you've a GPU, try the following installation instead.

```sh
> pip uninstall llama-cpp-python
> CMAKE_ARGS="-DLLAMA_CUBLAS=ON" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --verbose
```

## Roadmap

- [x] Chat and task modes for interactive or batch queries
- [x] Define new tasks with only a prompt, no code. See docs/samples directory
      for examples.
- [x] Markdown formatting for chat mode
- [ ] Command support in chat. E.g., logs, change model, copy, clear, etc.
- [ ] Discover prompts from user directory
- [ ] Manage prompts and create new interactively
- [ ] Download models and manage them
- [ ] Release v0.1
- [ ] Add [textfx](https://github.com/google/generative-ai-docs/tree/main/demos/palm/web/textfx)
- [ ] Add offline knowledge bases and RAG. See
      <https://library.kiwix.org/#lang=eng>

## License

Dedicated to the public domain with [CC0][].
We'll be delighted if this tool helps you positively 💖

[CC0]: https://creativecommons.org/publicdomain/zero/1.0/
