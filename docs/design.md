# Design

Notes on the design of this tool.

## Goals

- Simplify development with local large language models. Must be usable both as
  a CLI app and a library.
- Task oriented.
- Must support CPU based ggml models.
- Must be extensible for following dimensions. Preferably with configuration.
  - Newer models
  - Prompt formats
- Must support integration with other tools with in the ecosystem.
- Opinionated. Explicitly avoid dependency or feature bloat.

### Non Goals

We're not going to be build yet another index, or a semantic kernel like
`langchain`. We focus only on the language modeling aspect; not the storage or
anything else.

While these are true at the time of writing, we will be open to reconsider these
in the future.

- Training models are not supported. This is an inference library.
- GPU support is not available (due to lack of resource/testing environment).

### Better tools

We believe below tools are awesome and may be in the same bucket as this one.

- <https://github.com/simonw/llm>
- <https://github.com/go-skynet/LocalAI>

## Approach

### Mental model

We will build upon following concepts.

- **Models** are `ggml` binary files that can be generating text. We will use
  them for inference and performing tasks.
- **Prompt format** is the instruction set template for getting the model to do
  anything. A model's prompt format depends on its training data.
  - Example: Vicuna models follow `### Human:` and `### Assistant:` dialogue
    format.
  - Models and Prompt formats are coupled together.
  - A prompt format can be reused across multiple models.
- **Modes** of usage can be `chat` based where the interaction is modeled as
  a dialogue between 3 participants - SYSTEM, AI and USER; either single or
  multi turn. Or, the mode can be `instruct` where a model is prompted to carry
  out certain task.
  - Note that `chat` can be also used to carry out tasks.
  - Also, we can provide additional `context` in instruction mode.
  - Models can support one or both of these modes.
  - TODO: more clarification on where and how to use modes.
- **Tasks** are the primary interaction model for this app.
  - E.g., `myl <task>` will perform the task.
  - Task is the highest level of abstraction this library will provide.
  - Task will own and manage a prompt. E.g., it will understand the prompt
    parameters and will fill those at runtime.
  - Task will be stateless. It will not have any knowledge of stores etc. It can
    temporarily cache elements (TODO: memory?), but those will be ephemeral,
    will clean up with the session.
- **Prompt** is the actual instruction or context shared with a model to perform
  a Task. Obviously, Tasks and Prompts are coupled together.
  - A task can have multiple prompts.
  - A task can use multiple other tasks. E.g., chaining together.
  - A prompt has a template with well known parameters that can be replaced at
    runtime. E.g., `history` represents all the discussion in current session.
  - Every prompt will also publish its output or response format. A task will
    use these specifications.

**Scope of tasks**

Simple and self-sufficient tasks will be part of this library. They must have
a single objective. Think of the Unix philosophy. They can be composed with other
tasks or other apps to built larger tasks.

Tasks can use primitives like tokenize, parse, cleanup etc. This is a core
aspect of a natural language based tool.

Tasks and Prompts can be defined by the consumer app. This library will provide
abstractions to define and use those.

**Constraints**

Every task will _only_ support streaming. A consumer app can decide whether to stream,
or wait until all output is available.

### Example runs

Let's enumerate few example scenarios to ensure the mental model is sufficient.

**1. Summarize a text**

- Input: blob of text.
- Output: a summary of the provided text.
- Task can support various tweaks like summarize in bullets, or a paragraph.
  Optionally, extract keywords etc.
- E.g., `myl summarize --keywords <text blob>` or `cat essay.md | myl summarize`
- Implementation
  - Prompt can use zero shot or few shot mechanism.
  - Prompt response can be JSON. We can dynamically provide instruction to
    extract keyword. Or, keyword extraction can be a separate prompt.

**2. Q & A on a document**

- Inputs
  - Blob of text
  - Conversation history (optional)
  - Question
- Output
  - Answer
- Implementation
  - Use template variables for history to make sense of words like `it` in the
    question.
  - Will use DATA provided in prompt for answering.

**3. Q & A on a directory of files**

- Inputs: directory of files, and a question
- Output: answer
- Implementation
  - Implement as a separate app.
  - Semantic search to find answers and then use summarize task to create an
    answer.
