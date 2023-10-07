# Console abstraction

import io
import readline
import signal
import sys
from contextlib import contextmanager, redirect_stderr
from functools import lru_cache
from io import BytesIO, StringIO
from typing import Generator

from rich.console import Console
from rich.theme import Theme
from wurlitzer import pipes

theme = Theme(
    {
        "header": "",
        "message_prompt": "bold blue",
        "message_user": "blue",
        "message_ai": "bright_white",
        "message_footer": "dim white",
    }
)


@lru_cache(maxsize=None)
def get_console() -> Console:
    return Console(theme=theme)


class SignalContextManager:
    def __init__(self, signal_num, handler):
        self.signal_num = signal_num
        self.handler = handler
        self.original_handler = None

    def __enter__(self):
        self.original_handler = signal.signal(self.signal_num, self.handler)

    def __exit__(self, exc_type, exc_value, traceback):
        signal.signal(self.signal_num, self.original_handler)


@contextmanager
def capture_stderr() -> Generator[StringIO | BytesIO | int | None, None, None]:
    """A context manager that captures stderr for both python and
    c-functions."""
    stderr = io.StringIO()
    with redirect_stderr(stderr) as err, pipes(
        stdout=0, stderr=stderr, encoding="utf-8"
    ):
        yield err
