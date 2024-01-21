"""Services for the task command.

A task is stateless execution of a single instruction.
"""
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

from arey.ai import CompletionMetrics, ModelMetrics, combine_metrics
from arey.config import get_config
from arey.platform.console import capture_stderr
from arey.platform.llama import LlamaBaseModel
from arey.prompt import get_prompt, get_prompt_overrides

config = get_config()
model_path = config.chat.model.path
prompt_template = config.chat.model.template
model_settings = config.chat.settings
completion_settings = config.chat.profile

model: LlamaBaseModel = LlamaBaseModel(
    model_path=model_path, model_settings=model_settings
)
prompt_model = get_prompt(prompt_template)


@dataclass
class TaskResult:
    """Result of a task execution."""

    response: str
    metrics: CompletionMetrics
    finish_reason: Optional[str]
    logs: Optional[str]


@dataclass
class Task:
    """A task is a stateless script invocation with a prompt."""

    prompt_overrides: dict = field(default_factory=dict)
    result: Optional[TaskResult] = None


def create_task(prompt_file: Optional[str]) -> Tuple[Task, ModelMetrics]:
    """Create a task with given prompt file."""
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
    context = {
        "user_query": user_input,
        "chat_history": "",
    }
    prompt = prompt_model.get("task", context)

    ai_msg_text = ""
    usage_series = []
    finish_reason = ""
    with capture_stderr() as stderr:
        for chunk in model.complete(prompt, {"stop": prompt_model.stop_words}):
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
