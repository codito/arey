"""Unit tests for the configuration module."""

from unittest.mock import mock_open

import pytest
from pytest_mock import MockerFixture

from myl.config import get_config


@pytest.fixture
def config_file_empty(mocker: MockerFixture):
    mocker.patch("os.mkdir", return_value=None)
    mock_file = mocker.mock_open(read_data="")
    return mock_file


@pytest.fixture
def config_file_valid():
    with open("docs/config.yml", "r") as f:
        return f.read()


def test_get_config_throws_for_nonexistent_file(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.mkdir")

    with pytest.raises(Exception) as e:
        get_config()

    assert e.value.args[0].startswith("No config found")


def test_get_config_return_throws_for_empty_file(
    mocker: MockerFixture, config_file_empty
) -> None:
    mocker.patch("builtins.open", config_file_empty)

    with pytest.raises(Exception) as e:
        get_config()

    # "chat" key is not found in empty dictionary
    assert e.value.args[0].startswith("chat")


def test_get_config_return_config_for_valid_schema(
    mocker: MockerFixture, config_file_valid
):
    mocker.patch("builtins.open", mock_open(read_data=config_file_valid))
    config1 = get_config()

    config2 = get_config()

    assert config1 is not None
    assert config1 == config2
    assert len(config1.profiles) == 3
