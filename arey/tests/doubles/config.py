"""Test configuration."""

import os

DUMMY_MODELS_DIR = "~/models"
DUMMY_7B_MODEL = "~/models/dummy_model.gguf"


def get_dummy_config(fs) -> str:
    """Get the dummy test config.

    Args:
    ----
        fs: pyfakefs object

    """
    fs.create_dir(os.path.expanduser(DUMMY_MODELS_DIR))
    fs.create_file(os.path.expanduser(DUMMY_7B_MODEL))

    fs.pause()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "test_config.yml"), "r") as f:
        fs.resume()
        return f.read()
