from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from myl.ai import CompletionMetrics


class SenderType(Enum):
    SYSTEM = 1
    AI = 2
    USER = 3


@dataclass
class MessageContext:
    prompt: str
    finish_reason: Optional[str]
    metrics: CompletionMetrics


@dataclass
class Message:
    text: str
    timestamp: int  # unix timestamp
    sender: SenderType
    context: Optional[MessageContext]


@dataclass
class ChatContext:
    pass


@dataclass
class Chat:
    messages: List[Message] = field(default_factory=list)
    context: ChatContext = field(default_factory=ChatContext)
