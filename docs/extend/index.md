# Arey Extensions

You can extend `arey` to customize various use cases.

1. **Persona** can help use the LLM in a specific behavior or style. A persona
   can use any of the below extensions. See [persona] for a guide to create, or
   use characters.
2. **Tools** can perform an external task. E.g. get latest weather, calculate an
   expression, search. Please refer to [tools] for more guidance.
3. **Variables** are tokens that allow the LLM to retrieve specific information
   at runtime from various tools. E.g., refer to the shell history, or browser
   tab. Variables are exposed by the tools. We discuss this in [variables].
4. **Knowledge base** provides the LLM with external domain-specific and
   up-to-date knowledge on anything. E.g., Wikipedia or personal notes. More
   details are in [knowledge] guide.

[persona]: ./persona.md
[tools]: ./tools.md
[variables]: #
[knowledge]: #

## Examples

| Mechanism      | How to use             | Example                          |
| -------------- | ---------------------- | -------------------------------- |
| Persona        | @character or /persona | @socrates advise on <situation>  |
| Tool           | !tool or /tool         | !calc 2+3, or !weather           |
| Variables      | $source or /vars       | Summarize ${browser.tab} content |
| Knowledge base | #source or /kb         | Define stoicism from #notes      |
