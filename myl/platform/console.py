# Console abstraction

import readline
import signal
from functools import lru_cache

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
