"""Configuration for myl."""
import os
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict

import yaml

DEFAULT_DATA_DIR = os.path.expanduser(
    "~/.local/share/myl" if os.name == "posix" else "~/.myl"
)
DEFAULT_CONFIG_DIR = os.path.expanduser(
    "~/.config/myl" if os.name == "posix" else "~/.myl"
)


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

    model: ModelConfig
    profile: ProfileConfig
    settings: Dict


@dataclass
class Config:
    """Myl Configuration."""

    models: Dict[str, ModelConfig]
    profiles: Dict[str, ProfileConfig]
    chat: ChatConfig

    @classmethod
    def from_dict(cls, config: dict):
        """Create a configuration from dictionary."""
        models = {
            key: ModelConfig(val["path"], val["template"], val.get("type", "llama2"))
            for key, val in config.get("models", {}).items()
        }
        profiles = {
            key: ProfileConfig(**val) for key, val in config.get("profiles", {}).items()
        }

        chat = ChatConfig(
            models[config["chat"]["model"]],
            profiles[config["chat"]["profile"]],
            config["chat"]["settings"],
        )
        return cls(models, profiles, chat)


def _make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def _get_config_dir():
    base_dir = os.environ.get("XDG_CONFIG_HOME")
    config_dir = os.path.join(base_dir, "myl") if base_dir else DEFAULT_CONFIG_DIR
    _make_dir(config_dir)
    return config_dir


def get_config() -> Config:
    """Get the app configuration if available."""
    if getattr(get_config, "config", None):
        return get_config.config

    config_file = os.path.join(_get_config_dir(), "myl.yml")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            try:
                get_config.config = Config.from_dict(yaml.safe_load(f) or {})
                return get_config.config
            except Exception as _:
                raise
    raise Exception(f"No config found at '{config_file}'. Please run `myl setup`.")


def get_asset_dir(suffix: str = "") -> str:
    """Get path of the assets directory.

    params:
        suffix (str): suffix directory name to append
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "data", suffix)


def get_asset_path(asset_name: str) -> str:
    """Get the full path to included asset.

    params:
        asset_name (str): relative path to the asset. E.g., prompts/alpaca.yml
    """
    dir_path = get_asset_dir()
    file_path = os.path.join(dir_path, asset_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Asset file {asset_name} not found. Resolved path: {file_path}"
        )
    return file_path
