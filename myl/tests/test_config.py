"""Unit tests for the configuration module."""

import pytest
from pytest_mock import MockerFixture

from myl.config import get_config


@pytest.fixture
def config_file_empty(mocker: MockerFixture):
    mocker.patch("os.mkdir", return_value=None)
    mock_file = mocker.mock_open(read_data="")
    return mock_file


def test_get_config_return_empty_dict_for_nonexistent_file(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)

    config = get_config()

    assert config == {}


def test_get_config_return_empty_dict_for_empty_file(
    mocker: MockerFixture, config_file_empty
) -> None:
    mocker.patch("builtins.open", config_file_empty)

    config = get_config()

    assert config == {}
