# Console abstraction

import readline
import signal
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from os import devnull

from rich.console import Console
from rich.theme import Theme

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
def suppress_stderr():
    """A context manager that redirects stderr to devnull."""
    # See https://stackoverflow.com/a/52442331
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err:
            yield (err)
