# Configuration for myl
import os

import yaml


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
    if not getattr(get_config, "config", None):
        config_file = os.path.join(_get_config_dir(), "myl.yml")
        get_config.config = yaml.safe_load(open(config_file, "r"))
    return get_config.config
