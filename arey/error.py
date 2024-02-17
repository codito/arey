"""Error routines for Arey."""
from typing import Union, Literal


class AreyError(Exception):
    """Error in Arey execution."""

    category: Union[Literal["config"], Literal["template"], Literal["system"]]
    message: str

    def __init__(
        self,
        category: Union[Literal["config"], Literal["template"], Literal["system"]],
        message: str,
    ):
        """Create an instance of AreyError with category and message."""
        self.category = category
        self.message = message
        super().__init__(message)
