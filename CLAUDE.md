# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

See @README.md for an overview.

## Development Commands

### Building

- `cargo build` - Build the workspace
- `cargo build --release` - Build optimized release binaries
- `cargo run --bin arey` - Run the main arey binary
- `cargo run --bin tool_weather` - Run specific tools (e.g., weather tool)

### Testing

- `cargo test` - Run all tests in the workspace
- `cargo test --package arey-core` - Run tests for core crate
- `cargo test test_name` - Run a specific test function
- `cargo test test_name -- --exact` - Run exact single test without filtering
- `cargo nextest run` - Use nextest for faster parallel testing (if installed)

### Code Quality

- `cargo fmt --all` - Format code with rustfmt
- `cargo clippy --all-targets --all-features -- -D warnings` - Run clippy linter
- `cargo check` - Quick typecheck without full build

## Architecture

See @docs/design.md for design details.

## Development Guidance

See @CONVENTIONS.md for code style, conventions, and development philosophy.

- **Rust Edition**: 2024
- **Imports**: Group by std, external crates, internal modules. Use `use crate::mod;` for internal. No wildcards.
- **Formatting**: Use rustfmt defaults. Run `cargo fmt --all` after your changes
  are complete.
- **Types**: Be explicit. Prefer `Result<T, anyhow::Error>` for fallible functions
- **Error Handling**: Use `anyhow` for contextful errors. Propagate with `?`
- **Async**: Use `async fn` with `tokio::main`, `async_trait` for traits
- **Documentation**: `///` doc comments for public items, examples in tests
- **Testing**: Unit tests in `#[cfg(test)]` modules, `#[tokio::test]` for async
- **Logging**: Use tracing, never log secrets
- **Security**: Validate inputs with regex/serde
