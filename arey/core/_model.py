"""An AI model representation."""

from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelProvider = Literal["gguf", "openai", "ollama"]

ModelCapability = Literal[
    "completion",  # support text completions
    "embedding",  # support text embedding generation
    "chat",
    "code",
    "math",
    "tools:openai",  # support openai style function calls
    "tools:llama",  # support llama 3.x style function calls
]


class ModelConfig(BaseModel):
    """Configuration for a large language model."""

    name: str
    """Model name. Required."""

    type: ModelProvider
    """Model type. Required."""

    capabilities: list[ModelCapability] = ["completion"]
    """Capabilities of the model."""

    settings: dict[str, Any] = Field(default_factory=dict)
    """Provider specific settings."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def update_settings(self) -> Self:
        if self.__pydantic_extra__:
            self.settings = self.__pydantic_extra__
        return self


@dataclass
class ModelMetrics:
    """Metrics for the model."""

    init_latency_ms: float
