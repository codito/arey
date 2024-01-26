"""Unit tests for the configuration module."""

import os
import pytest
from pytest_mock import MockerFixture

from arey.config import create_or_get_config_file, get_config


@pytest.fixture(scope="session")
def test_dir(tmpdir_factory):
    tmp = tmpdir_factory.mktemp("arey").join("arey")
    return tmp


@pytest.fixture
def config_file_not_exist(test_dir, mocker: MockerFixture):
    mocker.patch("arey.config.get_config_dir", return_value=test_dir)
    return test_dir


@pytest.fixture
def config_file_empty(test_dir, mocker: MockerFixture):
    mocker.patch("arey.config.get_config_dir", return_value=test_dir)
    mocker.patch("os.mkdir", return_value=None)
    mocker.patch("os.path.exists", return_value=True)
    mock_file = mocker.mock_open(read_data="")
    return mock_file, test_dir


@pytest.fixture
def config_file_valid(test_dir, mocker: MockerFixture):
    mocker.patch("arey.config.get_config_dir", return_value=test_dir)
    mocker.patch("os.path.exists", return_value=True)
    config_path = os.path.join(test_dir, "arey.yml")
    with open("arey/data/config.yml", "r") as f:
        return f.read(), test_dir, config_path


def test_create_or_get_config_file_when_exists(config_file_valid):
    _, _, test_file = config_file_valid

    exists, file = create_or_get_config_file()

    assert exists is True
    assert file == test_file


def test_create_or_get_config_file_when_not_exist(config_file_not_exist, mocker):
    mocker.patch("arey.config.open", mocker.mock_open())
    test_file = os.path.join(config_file_not_exist, "arey.yml")

    exists, file = create_or_get_config_file()

    assert exists is False
    assert file == test_file


def test_get_config_throws_for_nonexistent_file(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.mkdir")

    with pytest.raises(Exception) as e:
        get_config()

    assert e.value.args[0]


def test_get_config_return_config_for_valid_schema(
    mocker: MockerFixture, config_file_valid
):
    test_data, _, _ = config_file_valid
    mocker.patch("arey.config.open", mocker.mock_open(read_data=test_data))
    config1 = get_config()

    config2 = get_config()

    assert config1 is not None
    assert config1 == config2
    assert len(config1.profiles) == 3
