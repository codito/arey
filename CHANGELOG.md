# Changelog

## Unreleased

## [0.1.0](https://github.com/codito/arey/compare/v0.0.6...v0.1.0) (2025-06-22)


### Features

* add ask command to arey-rust ([#6](https://github.com/codito/arey/issues/6)) ([1e841cf](https://github.com/codito/arey/commit/1e841cf4eeb2252f795578b351693c02a36fbbfa))
* add binary target entrypoint ([b977685](https://github.com/codito/arey/commit/b977685b9d6b172580fdf3a63ab248d47483457f))
* Add cancellation token to completion models ([5dd9f59](https://github.com/codito/arey/commit/5dd9f594a2d5cded20d2e3bbe8275e6676bd3a79))
* Add chat UI with spinner, Ctrl+C interrupt, and completion stats ([197f13d](https://github.com/codito/arey/commit/197f13d69e975e14419377c8a0612f9a29261160))
* Add Ctrl+D support to quit chat session ([6c3a4fa](https://github.com/codito/arey/commit/6c3a4fa7be32543354a733aa87e7f0484402a169))
* add Rust config module ([a02bf30](https://github.com/codito/arey/commit/a02bf30e67bee4e7d673ffe52002a3d405d66d05))
* Enable live reloading of play files ([29ce3c1](https://github.com/codito/arey/commit/29ce3c10c1b64f8582c79a91be9416928b4a7fa3))
* Implement cancellation for chat completion streams ([4054b72](https://github.com/codito/arey/commit/4054b72f893ee2e81517b1b5c047c065cafba747))
* Implement cancellation for chat stream ([2c18213](https://github.com/codito/arey/commit/2c18213a0ede5ea240e205767e0713749aa477cf))
* Implement chat commands and raw response logging ([147106f](https://github.com/codito/arey/commit/147106f9e477d2d1b8398764f5ae41c5db73f580))
* Implement chat message context, metrics, and system prompt ([97b43d9](https://github.com/codito/arey/commit/97b43d98dd0d956e9966c893c8e6fff035727f30))
* Implement chat message history ([8c6a99c](https://github.com/codito/arey/commit/8c6a99ca376d9c3e4ec2c2a5a3b4a3c9cb838025))
* Implement CLI structure with ask, chat, and play commands ([8153e75](https://github.com/codito/arey/commit/8153e75faa0bd4dde41f554ec74eab6fa5259bdf))
* Implement command autocomplete and contextual help system ([354045a](https://github.com/codito/arey/commit/354045a247ca06e893b8a26a2a9714adc1978627))
* Implement CompletionMetrics::combine for stream accumulation ([33d18ed](https://github.com/codito/arey/commit/33d18edb955228895a85563d3a50d0b843d0d763))
* Implement core chat module in Rust ([6f2e6b7](https://github.com/codito/arey/commit/6f2e6b734b7c2ed2650af0ad6e142fe5cabf7881))
* Implement OpenAI completion with streaming in Rust ([33db30d](https://github.com/codito/arey/commit/33db30d14037dd9e8e1ad5bd05beacd8dc657b79))
* Implement OpenAI raw output logging ([dff4c32](https://github.com/codito/arey/commit/dff4c329bfe051f8f6705a4299d9be86227edf74))
* Implement play command with file watching ([9db7e83](https://github.com/codito/arey/commit/9db7e83ccd78b0c33f05ef66bf9ce4cc2c951e33))
* Make ModelProvider enum serialize and display as lowercase ([32977e5](https://github.com/codito/arey/commit/32977e5cf217575bf414bf4e246b07c245e2407e))
* migrate platform module to Rust ([0dc2e7a](https://github.com/codito/arey/commit/0dc2e7a3768de50c99eae88abacf8273ddd570ad))
* port core model and completion modules from Python ([0fc90d7](https://github.com/codito/arey/commit/0fc90d78669faa7793b1f2d1e2d13555fd63846b))
* use local models via llama.cpp ([#5](https://github.com/codito/arey/issues/5)) ([1d316fb](https://github.com/codito/arey/commit/1d316fb643c374ddad43415e46ee1898a0ff7491))
* wip: make templates optional and use chat completion. ([ae1766d](https://github.com/codito/arey/commit/ae1766da48646982d246f8205d4b3529436f6158))


### Bug Fixes

* Add `StreamExt` import for stream `next()` method; remove unused imports ([e5030dc](https://github.com/codito/arey/commit/e5030dc2622e715f877e2c59b7e62a8af47aba45))
* Add clap dependency and correct HashMap::get key type ([d448382](https://github.com/codito/arey/commit/d44838297495d600f86d9811ccd85fcec1d584c7))
* Add content to mock stream to prevent test hang ([467e36a](https://github.com/codito/arey/commit/467e36a36279882c0860c4ea5348148f500fa91c))
* add temp named temp file for play ([47623b8](https://github.com/codito/arey/commit/47623b88d0f22a49dd393b89ea90bfcd7683322d))
* add uv to ci ([3f284d2](https://github.com/codito/arey/commit/3f284d23ced277d7e89d58a577a4d60b8ff9a4d9))
* Address OpenAI module compilation errors and type mismatches ([50d4ff2](https://github.com/codito/arey/commit/50d4ff22eb50cc2f6522918b1d2a779bfd9dd59a))
* allow model name override via cli. ([19fcc74](https://github.com/codito/arey/commit/19fcc74c53d8be4633885356f8766f6b3d1c9fa5))
* build warnings ([07a4277](https://github.com/codito/arey/commit/07a42778aef8260b321b9cc45188316af95ccf07))
* Clone stop_flag for cancellation task ([f8438b0](https://github.com/codito/arey/commit/f8438b0ff4583e7671e80da8a8d906d058ff4883))
* Correct HashMap lookup for String keys using as_str() ([56fe4b4](https://github.com/codito/arey/commit/56fe4b4cd7e07053ec23b9e7cf287a6219a61d39))
* Correct LLM model return type for trait objects and remove unused import ([ea7e4c6](https://github.com/codito/arey/commit/ea7e4c6bf48525a6645ea3cfa6a29631a32f6f2a))
* Correct model provider matching, Markdown AST, and stream handling ([a9ca0b7](https://github.com/codito/arey/commit/a9ca0b70e7aeb6c13659a555368ef3cf63037831))
* Correct OpenAI streaming SSE parsing and latency metrics ([e2e52a3](https://github.com/codito/arey/commit/e2e52a3dd255c4a363d91fae2fc6fc95c08a523f))
* Correct stream handling in `get_response` and remove unused import ([0ac88a5](https://github.com/codito/arey/commit/0ac88a529bc65ea286c90f22f33c47ccf4978b51))
* Correct XDG_CONFIG_HOME path in tests ([8ed7125](https://github.com/codito/arey/commit/8ed71255231de88692939cc6d501a27bbaf72d59))
* Correctly apply `?` operator to stream result ([3d1003f](https://github.com/codito/arey/commit/3d1003f2fe0c5505834e511994eebf172da95125))
* draft play migration ([1c05b64](https://github.com/codito/arey/commit/1c05b64661778f7ce1cef6519514544de8d3bab0))
* Enable mutable chat_ref for stream truncation ([45834f1](https://github.com/codito/arey/commit/45834f13a672d234f1049b05e4310309943aba0f))
* Enable streaming for OpenAI API requests ([a6ae54b](https://github.com/codito/arey/commit/a6ae54bd5ebfd93d0c87519c9ddb9c121f48b3db))
* Ensure stream completes naturally after metrics ([8e0ca83](https://github.com/codito/arey/commit/8e0ca833ae7bc39285ee09db77433cba6cc69060))
* Fix OpenAI integration compilation errors ([68c4fe1](https://github.com/codito/arey/commit/68c4fe1a38a4eec08eca9638de20e07b53738036))
* Fix OpenAI mock server setup and SSE response format in tests ([dc8be77](https://github.com/codito/arey/commit/dc8be77bbf5b14a3121ecbc7e64ce386754e6f06))
* Fix OpenAI mock stream body serialization in test ([1c02568](https://github.com/codito/arey/commit/1c025684ef2dbdbeee7c6a9d64e9995818290e4d))
* Fix OpenAI streaming implementation and error handling ([de5d03e](https://github.com/codito/arey/commit/de5d03ec3502e14745e3df3fe3af3d0150176fb2))
* Fix OpenAI test stream content type header ([54a3a98](https://github.com/codito/arey/commit/54a3a98da6d9ec88c41a29119cc2c51fc80ea91e))
* Fix test compilation errors and import issues ([c343d69](https://github.com/codito/arey/commit/c343d69fc8fa1bd8fd4da3cc0796b1773068377d))
* fixed single chat ([6932e8d](https://github.com/codito/arey/commit/6932e8d97451e20a4d85044715a8f9088029aefd))
* Handle stream result explicitly in get_response ([4775d0e](https://github.com/codito/arey/commit/4775d0e2824095378262847c314189cc3a29d1d0))
* Handle XDG_CONFIG_HOME gracefully in config tests ([cf0714d](https://github.com/codito/arey/commit/cf0714d8cea6c8daea1abea5c0eeb19ebc4dd748))
* Implement Default for CompletionMetrics, initialize completion_runs to 1 ([d594ef9](https://github.com/codito/arey/commit/d594ef965fee4dcb8ee9c6212615728e36d42d44))
* Make `create_missing` parameter concrete for type inference ([9298ed3](https://github.com/codito/arey/commit/9298ed353812571166cdf927a39f12de2e903be9))
* Make Ctrl-C instantly cancel ongoing completions ([d6329e8](https://github.com/codito/arey/commit/d6329e822eb50e769940c3911f60e64e8dcffdb9))
* message history ([f9dead9](https://github.com/codito/arey/commit/f9dead90bd503023a71658eee8fa7d4c0e8a908c))
* move tests to top-level and update imports ([aec7892](https://github.com/codito/arey/commit/aec78925969339c2c9d13cf922756212402b8f6c))
* openai tests ([37bb052](https://github.com/codito/arey/commit/37bb052079ec1045e5ff0a61e270dc0eb257bf6f))
* Pin Ctrl-C future to resolve E0277 and clean up warnings ([ca28eac](https://github.com/codito/arey/commit/ca28eacfb5876bd2fa41dfdbb5710c81658e26a1))
* play command and hang in completion. Add streaming token usage. ([6bcf84c](https://github.com/codito/arey/commit/6bcf84c9cd6270c1948bd8bccde9c0abe577af3c))
* play error scenarios ([28a41ab](https://github.com/codito/arey/commit/28a41abeb259950cf272764eda9f89fc5b627366))
* Prevent deadlock by accumulating metrics chunks ([7758396](https://github.com/codito/arey/commit/7758396c5997e04f42337b1f6facc1b6facad1c5))
* Prevent test hang by closing mock connection and sending DONE ([4d53742](https://github.com/codito/arey/commit/4d537424b4381de7d77e0589c5796719bef96673))
* read api_key from env var for openai compatible server. ([dd37cb6](https://github.com/codito/arey/commit/dd37cb6a6629a23e4b27fa3e189cefff69a746e4))
* remove Arc and locks for models in chat ([45b853a](https://github.com/codito/arey/commit/45b853a2ff9085f2eea09ed824ce7282c2cb41e8))
* Remove redundant Default derive from CompletionMetrics ([c1d5dca](https://github.com/codito/arey/commit/c1d5dca132fa30e107b649fe19a5f46ba5530aa5))
* Resolve config ownership and silence warnings ([fde7b14](https://github.com/codito/arey/commit/fde7b14d0fbe8c38a1f88ba8880d8f121e305898))
* Resolve lifetime and borrowing issues in get_response and run_once ([d9d0a90](https://github.com/codito/arey/commit/d9d0a900dfb39dd2bc3562d3a15774e7e8398603))
* Resolve stream ownership and duplicate import issues ([58748d4](https://github.com/codito/arey/commit/58748d45d2b0f2b2cc300a3f0e1c1afd90c293a9))
* Resolve type, ownership, and import issues across modules ([07d10a4](https://github.com/codito/arey/commit/07d10a4a07b3e335bdc123526717cf4f5368affc))
* resolve YAML deserialization conflict by removing serde flatten ([8291d6d](https://github.com/codito/arey/commit/8291d6d1b5d453d5f934b42f888be8f6bff98342))
* Respect XDG_CONFIG_HOME for config directory ([14fe205](https://github.com/codito/arey/commit/14fe205ea91e3142f79ecb9e08862b2c59363f9d))
* summarize example test with Gemma 2 2B. ([a0414f2](https://github.com/codito/arey/commit/a0414f2bceb59568fcbdb5db948678e030308094))
* tests ([38672c3](https://github.com/codito/arey/commit/38672c3f82df9f6801918a8356f6231f3906c41c))
* types for console and wrapper funcs ([0dfe693](https://github.com/codito/arey/commit/0dfe693e05a205d63e32698cfce3056809a9ebed))
* Use insert_header for mock response ([9d7fd76](https://github.com/codito/arey/commit/9d7fd76f78dfdeba2017fdcdcf56ba9eda1d356e))
* Use string literal for Content-Type in OpenAI mock tests ([dd7b814](https://github.com/codito/arey/commit/dd7b8145e40685cbdf974f99e93224029fe5dd48))
* Wrap std::env calls in unsafe blocks for config tests ([3347f50](https://github.com/codito/arey/commit/3347f50bd393609775d18fa5bf0d45a7162f317f))
* Wrap unsafe env ops in tests and remove unused import ([a3cd849](https://github.com/codito/arey/commit/a3cd849d843ea487dfa8976bb595978654a6729f))

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
