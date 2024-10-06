"""Core domain for the Arey app."""

from ._completion import (
    ChatMessage,
    CompletionMetrics,
    CompletionModel,
    CompletionResponse,
    SenderType,
    combine_metrics,
)
from ._error import AreyError
from ._model import ModelConfig, ModelMetrics

__all__ = [
    "AreyError",
    # Model contracts
    "ModelConfig",
    "ModelMetrics",
    # Completion API
    "ChatMessage",
    "CompletionModel",
    "CompletionResponse",
    "CompletionMetrics",
    "combine_metrics",
    # Chat
    "SenderType",
    "ChatMessage",
]
