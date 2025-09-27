# Arey Extensions

You can extend `arey` to customize various use cases.

1. **Agents** define AI personas with specific prompts, tool access, and behavioral
   settings. Agents are the primary way to customize LLM behavior. See [agent configuration](../config.md#agents) for details.
2. **Tools** can perform external tasks. E.g. get latest weather, calculate an
   expression, search. Please refer to [tools] for more guidance.
3. **Variables** are tokens that allow the LLM to retrieve specific information
   at runtime from various tools. E.g., refer to the shell history, or browser
   tab. Variables are exposed by the tools. We discuss this in [variables].
4. **Knowledge base** provides the LLM with external domain-specific and
   up-to-date knowledge on anything. E.g., Wikipedia or personal notes. More
   details are in [knowledge] guide.

> **Note**: The legacy [Persona](./persona.md) system has been replaced by the unified Agent system.

[tools]: ./tools.md
[variables]: #
[knowledge]: #

## Examples

| Mechanism      | How to use             | Example                          |
| -------------- | ---------------------- | -------------------------------- |
| Agent          | @agent_name            | @coder Write a function to...   |
| Tool           | !tool or /tool         | !calc 2+3, or !weather           |
| Variables      | $source or /vars       | Summarize ${browser.tab} content |
| Knowledge base | #source or /kb         | Define stoicism from #notes      |
