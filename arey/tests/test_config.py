"""Unit tests for the configuration module."""

import os
import pytest
from pytest_mock import MockerFixture

from arey.config import get_config
from arey.platform.assets import get_config_dir


@pytest.skip(allow_module_level=True)
@pytest.fixture
def config_file_empty(mocker: MockerFixture):
    mocker.patch("os.mkdir", return_value=None)
    mocker.patch("os.path.exists", return_value=True)
    mock_file = mocker.mock_open(read_data="")
    return mock_file


@pytest.fixture
def config_file_valid(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=True)
    with open("arey/data/config.yml", "r") as f:
        return f.read()


@pytest.fixture
def template_file_valid(mocker: MockerFixture):
    with open("arey/data/prompts/chatml.yml", "r") as f:
        return f.read()


def test_get_config_throws_for_nonexistent_file(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.mkdir")

    with pytest.raises(Exception) as e:
        get_config()

    assert e.value.args[0]


def test_get_config_creates_default_config_file(
    mocker: MockerFixture, config_file_valid, template_file_valid
) -> None:
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.mkdir")

    def fake_open(*args, **kwargs):
        global mock_config_open
        print(args)
        if args[0].endswith("arey.yml"):
            mock_config_open = mocker.mock_open(read_data=config_file_valid)
            return mock_config_open()
        return mocker.mock_open(read_data=template_file_valid)()

    mocker.patch("builtins.open", side_effect=fake_open)

    get_config()

    config_dir = get_config_dir()
    mock_config_open.assert_called_once_with(os.path.join(config_dir, "arey.yml"), "w")


def test_get_config_return_config_for_valid_schema(
    mocker: MockerFixture, config_file_valid, template_file_valid
):
    def fake_open(*args, **kwargs):
        if args[0].endswith("arey.yml"):
            return mocker.mock_open(read_data=config_file_valid)()
        return mocker.mock_open(read_data=template_file_valid)()

    mocker.patch("builtins.open", side_effect=fake_open)
    config1 = get_config()

    config2 = get_config()

    assert config1 is not None
    assert config1 == config2
    assert len(config1.profiles) == 3
