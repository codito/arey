"""Large language model abstraction."""
# pyright: strict, reportAny=false

from typing import Literal

from arey.core import CompletionModel, ModelConfig
from arey.platform._llama import LlamaBaseModel
from arey.platform._ollama import OllamaBaseModel
from arey.platform._openai import OpenAIBaseModel

ModelType = Literal["openai", "ollama", "gguf"]


def get_completion_llm(model_config: ModelConfig) -> CompletionModel:
    """Get a completion AI model."""
    if model_config.type == "ollama":
        return OllamaBaseModel(model_config)
    if model_config.type == "openai":
        return OpenAIBaseModel(model_config)

    return LlamaBaseModel(model_config)
