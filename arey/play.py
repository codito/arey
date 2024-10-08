"""Services for the play command.

Play reads the model details from a markdown file, and the prompt. It will watch
the file for any changes and run a completion for the prompt.
"""

import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import frontmatter  # pyright: ignore[reportMissingTypeStubs]

from arey.config import get_config
from arey.core import (
    AreyError,
    ChatMessage,
    CompletionMetrics,
    CompletionModel,
    ModelConfig,
    ModelMetrics,
    SenderType,
    combine_metrics,
)
from arey.platform.assets import get_asset_path
from arey.platform.console import capture_stderr
from arey.platform.llm import get_completion_llm

config = get_config()


@dataclass
class PlayResult:
    """Result of a task execution."""

    response: str
    metrics: CompletionMetrics
    finish_reason: str | None
    logs: str | None


@dataclass
class PlayFile:
    """A play file with model settings and a prompt."""

    file_path: str

    model_config: ModelConfig | None
    model_settings: dict[str, str]

    prompt: str
    completion_profile: dict[str, Any]

    output_settings: dict[str, str]

    model: CompletionModel | None = None
    result: PlayResult | None = None


@lru_cache(maxsize=1)
def _get_default_play_file() -> str:
    """Get the default play file template."""
    file_path = get_asset_path("play.md")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _create_file_if_not_exists(play_file_path: str) -> str:
    if play_file_path and os.path.exists(play_file_path):
        return play_file_path

    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix="arey_play", suffix=".md", delete=False
    ) as file:
        file.write(_get_default_play_file().encode())
        return file.name


def get_play_file(file_path: str) -> PlayFile:
    """Evaluate the provided play file and generate a response."""
    play_file_path = _create_file_if_not_exists(file_path)
    with open(play_file_path, "r", encoding="utf-8") as f:
        play_file = frontmatter.load(f)

    # FIXME validate settings
    model_config = config.models.get(cast(str, play_file.metadata["model"]), None)
    model_settings = cast(dict[str, Any], play_file.metadata.get("settings", {}))
    completion_profile = cast(dict[str, Any], play_file.metadata.get("profile", {}))
    output_settings = cast(dict[str, str], play_file.metadata.get("output", {}))
    return PlayFile(
        file_path=play_file_path,
        model_config=model_config,
        model_settings=model_settings,
        prompt=play_file.content,
        completion_profile=completion_profile,
        output_settings=output_settings,
    )


def load_play_model(play_file: PlayFile) -> ModelMetrics:
    """Load a model from play file."""
    model_config = play_file.model_config
    if model_config is None:
        raise AreyError(
            "config", "Please specify a valid model configuration in play file."
        )

    model_config.settings |= play_file.model_settings
    with capture_stderr():
        model: CompletionModel = get_completion_llm(model_config=model_config)
        model.load("")
        play_file.model = model
        return model.metrics


def get_play_response(play_file: PlayFile) -> Iterator[str]:
    """Run a task with user query."""
    model = play_file.model
    if not model:
        raise AreyError("system", "Model is empty.")

    completion_settings = play_file.completion_profile
    prompt = play_file.prompt
    ai_msg_text = ""
    usage_series: list[CompletionMetrics] = []
    finish_reason = ""
    with capture_stderr() as stderr:
        for chunk in model.complete(
            [ChatMessage(sender=SenderType.USER, text=prompt)], completion_settings
        ):
            ai_msg_text += chunk.text
            finish_reason = chunk.finish_reason
            usage_series.append(chunk.metrics)
            yield chunk.text

    play_file.result = PlayResult(
        response=ai_msg_text,
        metrics=combine_metrics(usage_series),
        finish_reason=finish_reason,
        logs=stderr.getvalue(),
    )
