# Changelog

## Unreleased

## v0.0.5 - 2024-02-17

- Feature: ollama support for ask, chat and play commands.

## v0.0.4 - 2024-01-22

This release focuses on fixing `arey play` command on Windows/WSL2 and multiple
performance fixes.

- Fix: don't reload model if settings remain unchanged.
- Fix: apply completion settings correctly including stop words.
- Fix: create config dir on Windows when absent.
- Fix: don't redirect stderr on Windows.

## v0.0.3 - 2024-01-21

- Initial pre-alpha release 🚀
- Chat mode for continuous learning with an AI model.
- Ask mode for a stateless single task.
- And a Play mode for prompt fine-tuning like a playground. Use your editor to
  update the prompt and `arey` will generate a response on save.
