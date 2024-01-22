"""Abstraction for various data and config assets in Arey."""
import os


DEFAULT_DATA_DIR = os.path.expanduser(
    "~/.local/share/arey" if os.name == "posix" else "~/.arey"
)
DEFAULT_CONFIG_DIR = os.path.expanduser(
    "~/.config/arey" if os.name == "posix" else "~/.arey"
)


def _make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_asset_dir(suffix: str = "") -> str:
    """Get path of the assets directory.

    params:
        suffix (str): suffix directory name to append
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(dir_path, "..", "data", suffix))


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


def get_config_dir():
    """Get arey config dir."""
    base_dir = os.environ.get("XDG_CONFIG_HOME")
    config_dir = os.path.join(base_dir, "arey") if base_dir else DEFAULT_CONFIG_DIR
    _make_dir(config_dir)
    return config_dir


def get_default_config() -> str:
    """Get default configuration template."""
    config_file = get_asset_path("config.yml")
    with open(config_file, "r", encoding="utf-8") as f:
        return f.read()
