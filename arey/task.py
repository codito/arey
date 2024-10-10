"""Services for the task command.

A task is stateless execution of a single instruction.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from arey.config import get_config
from arey.core import (
    ChatMessage,
    CompletionMetrics,
    CompletionModel,
    ModelMetrics,
    SenderType,
    combine_metrics,
)
from arey.platform.console import capture_stderr
from arey.platform.llm import get_completion_llm
from arey.prompt import get_prompt_overrides

config = get_config()
model_config = config.task.model
completion_settings = config.task.profile

model: CompletionModel = get_completion_llm(model_config)
prompt_model = None  # get_prompt(prompt_template) if prompt_template else None


@dataclass
class TaskResult:
    """Result of a task execution."""

    response: str
    metrics: CompletionMetrics
    finish_reason: str | None
    logs: str | None


@dataclass
class Task:
    """A task is a stateless script invocation with a prompt."""

    prompt_overrides: dict[str, str] = field(default_factory=dict)
    result: TaskResult | None = None


def create_task(prompt_file: str | None) -> tuple[Task, ModelMetrics]:
    """Create a task with given prompt file."""
    system_prompt = ""
    if prompt_model:
        token_overrides = (
            get_prompt_overrides(prompt_file).custom_tokens
            if prompt_file and os.path.exists(prompt_file)
            else {}
        )
        system_prompt = prompt_model.get_message("system", "", token_overrides)
    with capture_stderr():
        model.load(system_prompt)
    task = Task()
    return task, model.metrics


def run(task: Task, user_input: str) -> Iterator[str]:
    """Run a task with user query."""
    # FIXME
    # context = {
    #     "user_query": user_input,
    #     "chat_history": "",
    # }
    prompt = [
        ChatMessage(sender=SenderType.USER, text=user_input[0])
    ]  # prompt_model.get("task", context)

    ai_msg_text = ""
    usage_series: list[CompletionMetrics] = []
    finish_reason = ""
    with capture_stderr() as stderr:
        for chunk in model.complete(
            messages=prompt, settings={"stop": []}
        ):  # prompt_model.stop_words}):
            ai_msg_text += chunk.text
            finish_reason = chunk.finish_reason
            usage_series.append(chunk.metrics)
            yield chunk.text

    task.result = TaskResult(
        response=ai_msg_text,
        metrics=combine_metrics(usage_series),
        finish_reason=finish_reason,
        logs=stderr.getvalue(),
    )


def close(_task: Task):
    """Close a task and free the model."""
    if model:
        model.free()
