"""Completion models."""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

from arey.core._model import ModelConfig, ModelMetrics

SenderTypeLiteral = Literal["assistant", "user", "system"]


class SenderType(Enum):
    """Chat message sender."""

    SYSTEM = 1
    ASSISTANT = 2
    USER = 3

    def role(self) -> SenderTypeLiteral:
        """Convert the sender to role."""
        return cast(SenderTypeLiteral, self.name.lower())


@dataclass(kw_only=True)
class ChatMessage:
    """A chat message."""

    text: str
    sender: SenderType


@dataclass
class CompletionMetrics:
    """Metrics related to a single completion."""

    prompt_tokens: int
    """Number of tokens in the prompt for this completion."""

    prompt_eval_latency_ms: float
    """Time taken for prompt evaluation."""

    completion_tokens: int  # number of tokens in completions
    """Number of tokens in this completion."""

    completion_runs: int
    """Number of runs required to generate above tokens. Each run can generate
    more than one token.

    1 for streaming response. N for completed response.
    """

    completion_latency_ms: float
    """Time taken for this completion."""


@dataclass
class CompletionResponse:
    """Response from a generative ai model."""

    text: str
    finish_reason: str | None  # stop, length, none
    metrics: CompletionMetrics


class CompletionModel(ABC, metaclass=ABCMeta):
    """A generative AI model."""

    @property
    @abstractmethod
    def context_size(self) -> int:
        """Get context size for the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metrics(self) -> ModelMetrics:
        """Get metrics for the model."""
        raise NotImplementedError

    @abstractmethod
    def load(self, text: str) -> None:
        """Load the model with a warm up system prompt."""
        raise NotImplementedError

    @abstractmethod
    def complete(
        self, messages: list[ChatMessage], settings: dict[str, Any]
    ) -> Iterator[CompletionResponse]:
        """Create a completion for given messages."""
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for the given text."""
        raise NotImplementedError

    @abstractmethod
    def free(self) -> None:
        """Free any resources for the model."""
        raise NotImplementedError

    @staticmethod
    def validate_config(_config: ModelConfig) -> bool:
        """Validate the model configuration."""
        return True


def combine_metrics(usage_series: list[CompletionMetrics]) -> CompletionMetrics:
    """Join a series of completion metrics into one."""
    response_latency = 0
    response_tokens = 0
    for u in usage_series:
        response_latency += u.completion_latency_ms
        response_tokens += u.completion_tokens
    return CompletionMetrics(
        prompt_tokens=usage_series[-1].prompt_tokens,
        prompt_eval_latency_ms=usage_series[0].prompt_eval_latency_ms,
        completion_tokens=response_tokens,
        completion_runs=len(usage_series),
        completion_latency_ms=response_latency,
    )
