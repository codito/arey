# Models for AI
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import List


@dataclass
class ModelMetrics:
    init_latency_ms: float


@dataclass
class CompletionMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    time_to_first_token_ms: float


@dataclass
class CompletionResponse:
    text: str
    metrics: CompletionMetrics


class CompletionModel(ABC, metaclass=ABCMeta):
    @abstractproperty
    def metrics(self) -> ModelMetrics:
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
