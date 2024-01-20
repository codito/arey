"""Models for AI."""
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Optional, Iterator, List


@dataclass
class ModelMetrics:
    """Metrics for the model."""

    init_latency_ms: float


@dataclass
class CompletionMetrics:
    """Metrics related to a single completion."""

    """Number of tokens in the prompt for this completion."""
    prompt_tokens: int

    """Time taken for prompt evaluation."""
    prompt_eval_latency_ms: float

    """Number of tokens in this completion."""
    completion_tokens: int  # number of tokens in completions

    """Number of runs required to generate above tokens. Each run can generate
    more than one token.

    1 for streaming response. N for completed response.
    """
    completion_runs: int

    """Time taken for this completion."""
    completion_latency_ms: float


@dataclass
class CompletionResponse:
    """Response from a generative ai model."""

    text: str
    finish_reason: Optional[str]  # stop, length, none
    metrics: CompletionMetrics


class CompletionModel(ABC, metaclass=ABCMeta):
    """A generative AI model."""

    @abstractproperty
    def metrics(self) -> ModelMetrics:
        """Get metrics for the model."""
        raise NotImplementedError

    @abstractmethod
    def load(self, text: str):
        """Load the model with a warm up system prompt."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, text: str) -> Iterator[CompletionResponse]:
        """Create a completion for given text."""
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for the given text."""
        raise NotImplementedError


class EmbeddingModel(ABC, metaclass=ABCMeta):
    """An embedding AI model."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Count of dimensions supported by the model."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Create an embedding for given text."""
        raise NotImplementedError


def combine_metrics(usage_series: List[CompletionMetrics]) -> CompletionMetrics:
    """Join a series of completion metrics into one."""
    response_latency = 0
    response_tokens = 0
    for u in usage_series:
        response_latency += u.completion_latency_ms
        response_tokens += u.completion_tokens
    return CompletionMetrics(
        prompt_tokens=usage_series[0].prompt_tokens,
        prompt_eval_latency_ms=usage_series[0].prompt_eval_latency_ms,
        completion_tokens=response_tokens,
        completion_runs=len(usage_series),
        completion_latency_ms=response_latency,
    )
