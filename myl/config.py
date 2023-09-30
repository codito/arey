# Configuration for myl
import os
from dataclasses import dataclass

import yaml


@dataclass
class TaskConfig:
    """Configuration for a given task."""

    """Name of the model to use."""
    model: str

    """Completion profile to use. Controls the generation attributes."""
    profile: str

    """Inference settings for this model. E.g., threads, context size etc."""
    settings: dict


def _make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def _get_data_dir():
    base_dir = os.environ.get("XDG_DATA_HOME")
    data_dir = (
        os.path.join(base_dir, "myl")
        if base_dir
        else os.path.expanduser("~/.local/share/myl")
    )
    _make_dir(data_dir)
    return data_dir


def _get_config_dir():
    base_dir = os.environ.get("XDG_CONFIG_HOME")
    config_dir = (
        os.path.join(base_dir, "myl")
        if base_dir
        else os.path.expanduser("~/.config/myl")
    )
    _make_dir(config_dir)
    return config_dir


def get_config():
    """Gets the app configuration if available."""
    if not getattr(get_config, "config", None):
        config_file = os.path.join(_get_config_dir(), "myl.yml")
        config_content = open(config_file, "r")
        get_config.config = yaml.safe_load(config_content) or {}
    return get_config.config


def get_task_config(task: str) -> dict:
    """Gets the configuration of a given task, e.g., chat.

    Returns empty dictionary if settings are not available."""
    config = get_config()
    return config.get(task, {})
