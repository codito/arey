# myl

Set of tasks built on local language models.

> NOTE: Work in progress.
> Probably unusable at this time unless you're hacking on the code directly :)

## Goals

- Playground for fast development with language models
- Tasks should have a chat mode and an API mode for interop
- Opinionated, meant for personal workflows

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

## License

[CC0](https://creativecommons.org/publicdomain/zero/1.0/).
