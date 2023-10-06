# Models for AI
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelMetrics:
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
    text: str
    finish_reason: Optional[str]  # stop, length, none
    metrics: CompletionMetrics


class CompletionModel(ABC, metaclass=ABCMeta):
    @abstractproperty
    def metrics(self) -> ModelMetrics:
        raise NotImplementedError

    @abstractmethod
    def load(self, text: str):
        """Loads the model with a warm up system prompt."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, text: str) -> List[CompletionResponse]:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


class EmbeddingModel(ABC, metaclass=ABCMeta):
    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError
