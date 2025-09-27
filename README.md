# arey

> Arey (अरे, sanskrit) — ind. Interjection of calling.

Arey is a simple large language model playground in your terminal.

- ✨ Command line interface, built with Rust, runs everywhere.
- 🤖 Use any llama.cpp model, or an openai compatible endpoint.
- 💬 Chat with your favorite local models. CPU friendly 🍀
- 🙋 Ask anything to AI models with a single command.
- 📋 Supercharged prompt fine-tuning workflow ❤️ Edit your prompt in _any_ editor
  and `arey` will generate a completion on save.
- 🔓 No telemetry, completely private with local models, internet optional (if using tools).
- 🤖 Configurable agents for specialized tasks.
- 🔧 Extensible with tools for web search, file operations, and more.

See [Get Started](https://apps.codito.in/arey) or notes below for a quick guide.

https://github.com/codito/arey/assets/28766/6b886e49-6124-4256-84d9-20449c783a34

## Installation

**From source:**

```sh
git clone https://github.com/codito/arey.git
cd arey
cargo build --release
```

**Binary releases:**
Download the latest release from [GitHub Releases](https://github.com/codito/arey/releases).

**Package managers:**

- Cargo: `cargo install arey` or `cargo binstall arey`.

Windows troubleshooting notes are [here](docs/windows.md).

## Usage

```sh
❯ arey --help
Usage: arey [OPTIONS] COMMAND [ARGS]...

  Arey - a simple large language model app.

Options:
  -v, --verbose BOOLEAN  Show verbose logs.
  --help                 Show this message and exit.

Commands:
  ask   Run an instruction and generate response.
  chat  Chat with an AI model.
  play  Watch FILE for model, prompt and generate response on edit.
```

On the first run, `arey` will create a configuration file in following location:

- `~/.config/arey/arey.yml` for Linux or Mac systems.
- `~/.arey/arey.yml` for Windows.

Please update the `models` section in the config yml to your local model path.

### 1. Ask Arey something!

`arey ask "Who is Seneca? Tell me one of his teachings"`

### 2. Chat with Arey

`arey chat`

### 3. Use specialized Agents

`arey` supports specialized agents for different tasks:

```bash
# Ask a research agent to find information
arey ask "@researcher What are the latest developments in quantum computing?"

# Chat with a code expert
arey chat
> @coder Help me debug this Rust function: fn main() { println!("Hello"); }

# Use a creative writing assistant
arey ask "@writer Write a short story about a robot learning to paint"
```

### 4. Run Arey in play mode

Use to fine-tune a prompt in your editor while `arey` keeps completing your prompt on every save.

```sh
❯ arey play /tmp/arey_playzl9igj3d.md

Welcome to arey play! Edit the play file below in your favorite editor and I'll generate a
response for you. Use `Ctrl+C` to abort play session.

Watching `/tmp/arey_playzl9igj3d.md` for changes...

───────────────────────────────────── 2024-01-21 17:20:01 ──────────────────────────────────────
✓ Model loaded. 0.57s.

Life is short because it passes by quickly and can end at any moment. We should make the most of
our time here on earth and live a virtuous life according to stoicism.

◼ Canceled.

Watching `/tmp/arey_playzl9igj3d.md` for changes...
```

## Development

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## Configuration

Arey uses YAML configuration files:

- **Config file**: `~/.config/arey/arey.yml` (Linux/Mac) or `~/.arey/arey.yml` (Windows)
- **Agents directory**: `~/.config/arey/agents/` for custom agent definitions

The configuration includes model definitions, agent personas, and tool settings. See [docs/config.md](./docs/config.md) for detailed configuration options.

## License

GPLv3 or later
