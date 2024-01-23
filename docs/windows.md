# Windows usage

We recommend using `arey` in wsl2 for the best possible experience.

A brief troubleshooting guide for errors while using `arey` on windows.

### llama-cpp-python installation

**Symptom**

```sh
❯ pipx install arey
Fatal error from pip prevented installation. Full pip output in file:
    C:\Users\codito\AppData\Local\pipx\pipx\Logs\cmd_2024-01-22_12.27.53_pip_errors.log

pip failed to build package:
    llama-cpp-python

Some possibly relevant errors from pip install:
    error: subprocess-exited-with-error
    no such file or directory
    CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
    CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage

Error installing arey.
```

See <https://github.com/abetlen/llama-cpp-python#windows-notes>.

### I'm using WSL2 but the model load is extremely slow

Ensure that the `*.gguf` file is loaded from a native ext4 partition.

```sh
# You downloaded models can be in C:\Users\<user>\Downloads\ directory
# Copy those to native WSL2 file system

> mkdir ~/models
> cp /mnt/c/Users/codito/Downloads/openhermes-2.5-mistral-7b.Q5_K_M.gguf ~/models/

# Now use ~/models/openhermes-2.5-mistral-7b.Q5_K_M.gguf in ~/.config/arey/arey.yml
```

### Slow performance due to less memory and cpu

Create a `C:\Users\<user>\.wslconfig` with appropriate CPU and Memory configuration.
See <https://learn.microsoft.com/en-us/windows/wsl/wsl-config#wslconfig>.

Before creating below file, ensure WSL is shutdown with `wsl --shutdown`.

```sh
# Contents of .wslconfig
memory=32GB      # for a machine with 64GB ram, change based on your system
processors=16    # for a machine with 16 logical processors
swap=4GB
```

Save this file and start wsl.
