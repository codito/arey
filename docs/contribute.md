# Contribute

Arey is under active development. We're building this app under a permissive
license to enable usage anywhere and in any shape or form.

We gladly accept contributions from the community.

Please create [an issue](https://github.com/codito/arey/issues/new) to share
your feedback, any feature requests or bug reports.

Thank you ❤️

## Development notes

### Building from source

```sh
# Clone the repository
git clone https://github.com/codito/arey.git
cd arey

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run with debug output
cargo run --bin arey -- --verbose chat
```

### Code style and formatting

```sh
# Format code
cargo fmt --all

# Run linter
cargo clippy --all-targets --all-features -- -D warnings

# Type check
cargo check
```

### Running tests

```sh
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with nextest (if installed)
cargo nextest run
```

### GPU support

For CUDA support, ensure you have the appropriate CUDA toolkit installed and build with:

```sh
# The build system will automatically detect CUDA if available
cargo build --release --features cuda
```

### CPU optimization

For better performance with CPU-based models, ensure you have system BLAS libraries installed:

```sh
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# macOS
brew install openblas

# The build system will automatically use BLAS if available
cargo build --release
```
