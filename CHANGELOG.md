# Changelog

## Unreleased

## [0.1.0](https://github.com/codito/arey/compare/v0.0.6...v0.1.0) (2024-10-18)


### Features

* wip: make templates optional and use chat completion. ([ae1766d](https://github.com/codito/arey/commit/ae1766da48646982d246f8205d4b3529436f6158))


### Bug Fixes

* add uv to ci ([3f284d2](https://github.com/codito/arey/commit/3f284d23ced277d7e89d58a577a4d60b8ff9a4d9))
* allow model name override via cli. ([19fcc74](https://github.com/codito/arey/commit/19fcc74c53d8be4633885356f8766f6b3d1c9fa5))
* play error scenarios ([28a41ab](https://github.com/codito/arey/commit/28a41abeb259950cf272764eda9f89fc5b627366))
* read api_key from env var for openai compatible server. ([dd37cb6](https://github.com/codito/arey/commit/dd37cb6a6629a23e4b27fa3e189cefff69a746e4))
* summarize example test with Gemma 2 2B. ([a0414f2](https://github.com/codito/arey/commit/a0414f2bceb59568fcbdb5db948678e030308094))
* types for console and wrapper funcs ([0dfe693](https://github.com/codito/arey/commit/0dfe693e05a205d63e32698cfce3056809a9ebed))

## v0.0.6 - 2024-08-04

- Feature: initial support for openai compatible servers.
- Feature: new models - Gemma 2, Llama 3.1, Codegeex 4.
- Fix: add readline support, update to latest llama-cpp for newer models
  including Gemma2 and so on.

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
