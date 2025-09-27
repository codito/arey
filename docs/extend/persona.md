# Using a Persona (Legacy)

> **Note**: The persona system has been replaced by the unified [Agent](../../config.md#agents) system. This documentation is preserved for historical reference.

The legacy persona system allowed you to chat with specific characters. This functionality has been enhanced and replaced by the unified Agent system, which provides more powerful capabilities including:

- **Tool integration**: Agents can use tools like search and file operations
- **Runtime state management**: Agents can have session-specific overrides
- **Multi-source loading**: Agents can be loaded from multiple locations with precedence
- **Enhanced configuration**: More flexible profile and tool configuration

## Migration to Agents

To migrate from personas to agents:

1. **Move files**: Move persona files from `~/.config/arey/persona/` to `~/.config/arey/agents/`
2. **Update format**: Convert from persona format to agent format:

   **Legacy persona format:**
   ```yml
   name: "phily"
   character:
     system_prompt: |
       You're phily, a philosopher from the same time as Socrates, Plato and
       others. You're well known for your deep knowledge of stoicism.
   ```

   **New agent format:**
   ```yml
   # ~/.config/arey/agents/phily.yml
   name: "phily"
   prompt: |
     You're phily, a philosopher from the same time as Socrates, Plato and
     others. You're well known for your deep knowledge of stoicism.
   tools: []
   profile:
     temperature: 0.7
   ```

3. **Update usage**: Replace `/persona phily` with `@phily` in chat sessions

For more information about creating and using agents, see the [Agent Configuration](../../config.md#agents) documentation.
