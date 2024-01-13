"""Core data structures for the myl app."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from myl.ai import CompletionMetrics, ModelMetrics


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
