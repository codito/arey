"""Console platform abstraction."""

import io
import signal
from contextlib import contextmanager, redirect_stderr
from functools import lru_cache
from io import StringIO
from typing import Generator

from rich.console import Console
from rich.theme import Theme

theme = Theme(
    {
        "header": "",
        "message_prompt": "bold blue",
        "message_user": "blue",
        "message_ai": "bright_white",
        "message_footer": "dim white",
        "error": "bold red",
    }
)


@lru_cache(maxsize=None)
def get_console() -> Console:
    """Get a console instance."""
    return Console(theme=theme)


class SignalContextManager:
    """Context manager for console signals."""

    def __init__(self, signal_num, handler):
        """Create a signal context instance."""
        self.signal_num = signal_num
        self.handler = handler
        self.original_handler = None

    def __enter__(self):
        """Context manager enter implementation."""
        self.original_handler = signal.signal(self.signal_num, self.handler)

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit implementation."""
        signal.signal(self.signal_num, self.original_handler)


@contextmanager
def capture_stderr() -> Generator[StringIO, None, None]:
    """Capture stderr for both python and c-functions."""
    stderr = io.StringIO()
    try:
        from wurlitzer import pipes

        with redirect_stderr(stderr) as err, pipes(
            stdout=0, stderr=stderr, encoding="utf-8"
        ):
            yield err
    except Exception:
        # Not supported on Windows. Disable stderr capture.
        yield stderr
