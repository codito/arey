# Contribute

Arey is under active development. We're building this app under a public domain
license to enable usage anywhere and in any shape or form.

We gladly accept contributions from the community.

Please create [an issue](https://github.com/codito/arey/issues/new) to share
your feedback, any feature requests or bug reports.

Thank you ❤️

## Development notes

```sh
# Install arey locally in editable mode.
> pip install -e .
> pip install -e .\[test\] # optional, if you wish to run tests

# Install with samples dependency if you wish to run them
> pip install -e .\[samples\]
```

With OPENBLAS, loading time for models is much smaller and inference is about
4-5% faster. Here's how to install `llama-cpp-python` with OPENBLAS support:

```sh
> pip uninstall llama-cpp-python
> CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --verbose --no-cache
```

If you've a GPU, try the following installation instead.

```sh
> pip uninstall llama-cpp-python
> CMAKE_ARGS="-DLLAMA_CUBLAS=ON" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --verbose --no-cache
```
