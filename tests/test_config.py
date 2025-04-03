"""Unit tests for the configuration module."""
# pyright: basic

import os

import pytest
from pytest_mock import MockerFixture

from arey.config import create_or_get_config_file, get_config
from arey.platform.assets import DEFAULT_CONFIG_DIR

from .doubles.config import get_dummy_config

DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "arey.yml")


@pytest.fixture
def default_config_file(fs, mocker: MockerFixture):
    config_file_content = get_dummy_config(fs)
    mocker.patch("arey.config.get_default_config", return_value=config_file_content)


def test_create_or_get_config_file_when_exists(fs):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(DEFAULT_CONFIG_FILE)
    with open(DEFAULT_CONFIG_FILE, "w") as f:
        f.write("dummy content")

    exists, file = create_or_get_config_file()

    assert exists is True
    assert file == DEFAULT_CONFIG_FILE


def test_create_or_get_config_file_when_not_exist(fs, default_config_file):
    exists, file = create_or_get_config_file()

    assert exists is False
    assert file == DEFAULT_CONFIG_FILE


def test_get_config_throws_for_nonexistent_file(fs, mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.mkdir")

    with pytest.raises(Exception) as e:
        get_config()

    assert e.value.args[0]


def test_get_config_return_config_for_valid_schema(fs, default_config_file):
    config1 = get_config()

    config2 = get_config()

    assert config1 is not None
    assert config1 == config2
    assert len(config1.profiles) == 3


def test_get_config_throws_for_invalid_yaml(fs, mocker: MockerFixture):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(DEFAULT_CONFIG_FILE, contents="invalid yaml content")

    with pytest.raises(AreyError) as e:
        get_config()

    assert "Configuration is invalid" in e.value.args[0]


def test_get_config_throws_for_missing_models(fs, mocker: MockerFixture):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(DEFAULT_CONFIG_FILE, contents="profiles: {}\nchat: {}\ntask: {}")

    with pytest.raises(AreyError) as e:
        get_config()

    assert "`models` is not provided in configuration." in e.value.args[0]


def test_get_config_throws_for_missing_chat_section(fs, mocker: MockerFixture):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(
        DEFAULT_CONFIG_FILE,
        contents="""
models:
  dummy_model:
    name: dummy_model
task: {}
""",
    )

    with pytest.raises(AreyError) as e:
        get_config()

    assert "`chat` and `task` sections are not available in config file." in e.value.args[0]


def test_get_config_throws_for_missing_task_section(fs, mocker: MockerFixture):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(
        DEFAULT_CONFIG_FILE,
        contents="""
models:
  dummy_model:
    name: dummy_model
chat: {}
""",
    )

    with pytest.raises(AreyError) as e:
        get_config()

    assert "`chat` and `task` sections are not available in config file." in e.value.args[0]


def test_get_config_throws_for_invalid_model_reference(fs, mocker: MockerFixture):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(
        DEFAULT_CONFIG_FILE,
        contents="""
models:
  dummy_model:
    name: dummy_model
chat:
  model: non_existent_model
task:
  model: non_existent_model
""",
    )

    with pytest.raises(AreyError) as e:
        get_config()

    assert "Configuration is invalid" in e.value.args[0]


def test_get_config_throws_for_invalid_profile_reference(fs, mocker: MockerFixture):
    fs.create_dir(DEFAULT_CONFIG_DIR)
    fs.create_file(
        DEFAULT_CONFIG_FILE,
        contents="""
models:
  dummy_model:
    name: dummy_model
profiles:
  default_profile:
    temperature: 0.7
    repeat_penalty: 1.176
    top_k: 40
    top_p: 0.1
chat:
  model: dummy_model
  profile: non_existent_profile
task:
  model: dummy_model
  profile: non_existent_profile
""",
    )

    with pytest.raises(AreyError) as e:
        get_config()

    assert "Configuration is invalid" in e.value.args[0]
