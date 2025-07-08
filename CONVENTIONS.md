# Contribution Guide

`arey` is a simple LLM app for the terminal built on following principles.

1. Simple, clean code with minimal dependencies.
2. Small core layer with just enough extensibility.
3. Usable, reliable and performant.

## Repo structure

This is a `rust` monorepo with crates in `crates/` directory.

- `crates/core` is the domain logic for the app along with extension points.
  Support multiple LLM providers and capabilities like tools etc.
- `crates/tools-*` are the various tools. E.g., search, memory etc. Each tool
  can be used inline or as an independent MCP (model context protocol) server.
- `crates/arey` is the cli app. Builds on core and tools.

## Rules

Use the below rules for your code contributions in this repo.

### Coding

- Always prefer idiomatic rust for your changes.
- Focus on the task at hand. Never mix feature change and refactoring together.
- Smaller commits are better.
- Respect the existing structure and convention in the repo.

### Design

- You must respect DRY, SOLID and similar clean code practices.
- Do not introduce unnecessary dependencies.
- Code must be usable, correct and performant.

### Testing

- Always add unit tests for a change.
