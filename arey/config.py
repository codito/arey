"""Configuration for arey."""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, TypedDict, Tuple, Union, cast

import yaml

from arey.model import AreyError
from arey.platform.assets import get_config_dir, get_default_config


@dataclass
class ModelConfig:
    """Configuration for a given model."""

    path: str
    template: str
    type: Optional[str] = "llama2"


class ProfileConfig(TypedDict):
    """Configuration for a given profile."""

    temperature: float
    repeat_penalty: float
    top_k: int
    top_p: float


@dataclass
class ChatConfig:
    """Configuration for chat mode."""

    model_name: str
    model: ModelConfig
    profile: ProfileConfig
    settings: Dict = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for task mode."""

    model_name: str
    model: ModelConfig
    profile: ProfileConfig
    settings: Dict = field(default_factory=dict)


@dataclass
class Config:
    """Arey Configuration."""

    models: Dict[str, ModelConfig]
    profiles: Dict[str, ProfileConfig]
    chat: ChatConfig
    task: TaskConfig

    @classmethod
    def from_dict(cls, config: dict):
        """Create a configuration from dictionary."""
        from arey.prompt import has_prompt

        models = {
            key: ModelConfig(val["path"], val["template"], val.get("type", "llama2"))
            for key, val in config.get("models", {}).items()
        }
        profiles = {
            key: ProfileConfig(**val) for key, val in config.get("profiles", {}).items()
        }
        default_profile: ProfileConfig = {
            "temperature": 0.7,
            "repeat_penalty": 1.176,
            "top_k": 40,
            "top_p": 0.1,
        }

        if "chat" not in config or "task" not in config:
            raise AreyError(
                "config", "`chat` and `task` sections are not available in config file."
            )

        def _get_config(key: str) -> Union[ChatConfig, TaskConfig]:
            model_name = config[key].get("model", None)
            if not model_name or model_name not in models:
                raise AreyError(
                    "config", f"Section '{key}' must have valid `model` entry."
                )
            model = models[model_name]
            model.path = os.path.expanduser(model.path) if model.path else model.path
            if not os.path.exists(model.path):
                raise AreyError(
                    "config", f"Model '{model_name}' has invalid path: {model.path}."
                )
            if not has_prompt(model.template):
                raise AreyError(
                    "config",
                    f"Model '{model_name}' has invalid template: {model.template}.",
                )

            profile = profiles.get(
                config[key].get("profile", "default"), default_profile
            )
            settings = config[key]["settings"] if "settings" in config[key] else {}
            if key == "chat":
                return ChatConfig(model_name, model, profile, settings)
            return TaskConfig(model_name, model, profile, settings)

        chat = _get_config("chat")
        task = _get_config("task")
        return cls(models, profiles, cast(ChatConfig, chat), cast(TaskConfig, task))


def create_or_get_config_file() -> Tuple[bool, str]:
    """Get config file path if exists, create a default otherwise."""
    config_file = os.path.join(get_config_dir(), "arey.yml")
    if os.path.exists(config_file):
        return (True, config_file)

    # Create a default config, ask user to update the model
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(get_default_config())
    return (False, config_file)


def get_config() -> Config:
    """Get the app configuration if available."""
    config = getattr(get_config, "config", None)
    if config:
        return config

    _, config_file = create_or_get_config_file()
    with open(config_file, "r") as f:
        config = Config.from_dict(yaml.safe_load(f) or {})
        setattr(get_config, "config", config)
        return config
