"""Error routines for Arey."""

from typing import Literal


class AreyError(Exception):
    """Error in Arey execution."""

    category: Literal["config", "template", "system"]
    message: str

    def __init__(
        self,
        category: Literal["config"] | Literal["template"] | Literal["system"],
        message: str,
    ):
        """Create an instance of AreyError with category and message."""
        self.category = category
        self.message = message
        super().__init__(message)
