"""Core data structures for the arey app."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Literal

from arey.ai import CompletionMetrics, ModelMetrics


class SenderType(Enum):
    """Chat message sender."""

    SYSTEM = 1
    AI = 2
    USER = 3


@dataclass
class MessageContext:
    """Context associated with a single chat message."""

    prompt: str
    finish_reason: Optional[str]
    metrics: CompletionMetrics
    logs: str = ""


@dataclass
class Message:
    """A chat message."""

    text: str
    timestamp: int  # unix timestamp
    sender: SenderType
    context: MessageContext


@dataclass
class ChatContext:
    """Context associated with a chat."""

    metrics: Optional[ModelMetrics] = None
    logs: str = ""


@dataclass
class Chat:
    """A chat conversation between human and AI model."""

    messages: List[Message] = field(default_factory=list)
    context: ChatContext = field(default_factory=ChatContext)


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
