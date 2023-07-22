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

```

With OPENBLAS, loading time for models is much smaller and inference is about
4-5% faster. Here's how to install `llama-cpp-python` with OPENBLAS support:

```sh
> CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --verbose
```

## License

[CC0](https://creativecommons.org/publicdomain/zero/1.0/).
