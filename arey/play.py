"""Services for the play command.

Play reads the model details from a markdown file, and the prompt. It will watch
the file for any changes and run a completion for the prompt.
"""
from dataclasses import dataclass
import os
import tempfile
import frontmatter
from functools import lru_cache
from arey.ai import CompletionMetrics, ModelMetrics, combine_metrics
from arey.config import ModelConfig, get_config
from arey.model import AreyError
from arey.platform.assets import get_asset_path
from arey.platform.llama import LlamaBaseModel
from arey.platform.console import capture_stderr
from typing import Dict, Optional, Iterator, cast


config = get_config()


@dataclass
class PlayResult:
    """Result of a task execution."""

    response: str
    metrics: CompletionMetrics
    finish_reason: Optional[str]
    logs: Optional[str]


@dataclass
class PlayFile:
    """A play file with model settings and a prompt."""

    file_path: str

    model_config: ModelConfig
    model_settings: Dict

    prompt: str
    completion_profile: Dict

    model: Optional[LlamaBaseModel] = None
    result: Optional[PlayResult] = None


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
    model_config = config.models[cast(str, play_file.metadata["model"])]
    model_settings: dict = cast(dict, play_file.metadata.get("settings", {}))
    completion_profile: dict = cast(dict, play_file.metadata.get("profile", {}))
    return PlayFile(
        file_path=play_file_path,
        model_config=model_config,
        model_settings=model_settings,
        prompt=play_file.content,
        completion_profile=completion_profile,
    )


def load_play_model(play_file: PlayFile) -> ModelMetrics:
    """Load a model from play file."""
    model_config = play_file.model_config
    model_settings = play_file.model_settings
    with capture_stderr():
        model: LlamaBaseModel = LlamaBaseModel(
            model_path=model_config.path, model_settings=model_settings
        )
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
    usage_series = []
    finish_reason = ""
    with capture_stderr() as stderr:
        for chunk in model.complete(prompt, completion_settings):
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
