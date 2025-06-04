# Using a Persona

Persona allows you to chat with a specific character. E.g., the `thinker`
persona will deliberate carefully and evaluate the pros/cons for generating
a response.

## Quick start

In a `arey chat` session, you can pick a persona for the entire session using the
`/persona <persona name>` slash command. The conversation will use persona's
character, style and generation instructions.

You can also invoke a persona temporarily using the `@persona_name
<instruction>` input in the chat session.

TODO: screenshot

## Create a Persona

`arey` picks up personas defined in the [config] directory. E.g.,
`~/.config/arey/persona`.

Every persona is a simple `yaml` file.

```yml
# Save this to ~/.config/arey/persona/phily.yml
name: "name of the persona. E.g., phily"
character:
  system_prompt: |
    You're phily, a philosopher from the same time as Socrates, Plato and
    others. You're well known for your deep knowledge of stoicism.
```

`arey` will validate available persona on startup. An error loading a persona is
considered a warning. It will ignore the persona, and you will a note when you
try to use it.
