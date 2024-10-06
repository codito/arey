"""Configuration for arey."""

import os
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from arey.core import AreyError, ModelConfig
from arey.platform.assets import get_config_dir, get_default_config


class ProfileConfig(BaseModel):
    """Configuration for a given profile."""

    temperature: float
    repeat_penalty: float
    top_k: int
    top_p: float


_DEFAULT_PROFILE = ProfileConfig(
    temperature=0.7,
    repeat_penalty=1.176,
    top_k=40,
    top_p=0.1,
)


class ModeConfig(BaseModel):
    """Configuration for chat or task mode."""

    model: ModelConfig
    profile: ProfileConfig = Field(default=_DEFAULT_PROFILE)


class Config(BaseModel):
    """Arey Configuration."""

    models: dict[str, ModelConfig]
    profiles: dict[str, ProfileConfig]
    chat: ModeConfig
    task: ModeConfig

    @classmethod
    @field_validator("chat", "task", mode="before")
    def chat_and_task_must_be_valid(cls, values: dict[str, Any]):
        """Validate chat and task must be non-empty."""
        if "chat" not in values or "task" not in values:
            raise AreyError(
                "config", "`chat` and `task` sections are not available in config file."
            )

    @model_validator(mode="before")
    @classmethod
    def config_deserializer(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Deserialize model config."""
        # Fill in the model name if not explicitly provided
        if "models" not in values:
            raise AreyError("config", "`models` is not provided in configuration.")
        for k, _ in values["models"].items():  # pyright: ignore[reportAny]
            values["models"][k]["name"] = k

        # For chat and task config, let's fill the model config
        for mode in ["chat", "task"]:
            if (
                mode not in values
                or "model" not in values[mode]
                or isinstance(values[mode]["model"], dict)  # already a dict
            ):
                continue

            # Resolve the model
            model_name = cast(str, values[mode]["model"])
            values[mode]["model"] = values["models"][model_name]

            if (
                "profile" not in values[mode]
                or isinstance(values[mode]["profile"], dict)  # already a dict
            ):
                continue

            # Resolve the profile
            profile_name = cast(str, values[mode]["profile"])
            values[mode]["profile"] = values["profiles"][profile_name]
        return values


def create_or_get_config_file() -> tuple[bool, str]:
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
        return cast(Config, config)

    _, config_file = create_or_get_config_file()
    with open(config_file, "r", encoding="utf-8") as f:
        config_data: dict[str, Any] = yaml.safe_load(f) or {}
        try:
            config = Config(**config_data)  # pyright: ignore[reportAny]
        except ValidationError as e:
            raise AreyError("config", f"Configuration is invalid. Errors: {e.errors}")
        setattr(get_config, "config", config)
        return config
