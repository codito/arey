"""Large language model abstraction."""

from arey.ai import CompletionModel
from arey.platform._llama import LlamaBaseModel
from arey.platform._ollama import OllamaBaseModel


def get_completion_llm(model_config: dict, settings: dict) -> CompletionModel:
    """Get a completion AI model."""
    model_name = model_config["name"]
    model_path = model_config["path"]
    if model_config["type"] == "ollama":
        return OllamaBaseModel(model_name, settings)
    return LlamaBaseModel(model_path, settings)


def validate_config(model_config: dict) -> bool:
    """Validate the model configuration."""
    if model_config["type"] == "ollama":
        return OllamaBaseModel.validate_config(model_config)
    return LlamaBaseModel.validate_config(model_config)
