"""Ollama based completion model."""

import dataclasses
import time
from typing import Any, Iterator, cast, Mapping

from ollama import Client, Options

from arey.ai import (
    ChatMessage,
    CompletionMetrics,
    CompletionModel,
    CompletionResponse,
    ModelMetrics,
)
from arey.error import AreyError


@dataclasses.dataclass
class OllamaSettings:
    """Core model settings."""

    host: str = "http://localhost:11434"


class OllamaBaseModel(CompletionModel):
    """Base local completion model.

    Wraps over the ollama library.
    """

    client: Client
    _model_name: str
    _model_settings: OllamaSettings
    _model_ctx_size: int = 4096
    _metrics: ModelMetrics

    def __init__(self, model_name: str, model_settings: dict = {}) -> None:
        """Create an instance of local completion model."""
        self._llm = None
        self._model_name = model_name
        self._model_settings = OllamaSettings(**model_settings)

    @property
    def context_size(self) -> int:
        return 0

    @property
    def metrics(self) -> ModelMetrics:
        return self._metrics

    def _get_message_from_chat(self, message: ChatMessage) -> dict[str, str]:
        if message.sender.role() == "system":
            return {"role": "system", "content": message.text}
        if message.sender.role() == "user":
            return {"role": "user", "content": message.text}
        if message.sender.role() == "assistant":
            return {"role": "assistant", "content": message.text}
        raise AreyError("system", f"Unknown message role: {message.sender.role()}")

    def load(self, text: str) -> None:
        self.client = Client(**dataclasses.asdict(self._model_settings))
        # response = self.client.show(self._model_name)
        # self._model_ctx_size = response["parameters"]["num_ctx"]

        response = cast(
            Mapping[str, Any], self.client.generate(model=self._model_name, prompt=text)
        )

        load_latency_ms = round(response.get("load_duration", 0) / 1000, 2)
        self._metrics = ModelMetrics(init_latency_ms=load_latency_ms)

    def complete(
        self, messages: list[ChatMessage], settings: dict = {}
    ) -> Iterator[CompletionResponse]:
        # TODO: add chat completion support
        prev_time = time.perf_counter()
        completion_settings = {
            "num_predict": -1,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.1,
            "repeat_penalty": 1.176,
            "raw": True,  # do not preserve context in the server
        } | settings
        output = cast(
            Iterator[Mapping[str, Any]],
            self.client.chat(
                model=self._model_name,
                messages=[self._get_message_from_chat(m) for m in messages],
                options=self._get_options(completion_settings),
                stream=True,
            ),
        )

        prompt_token_count = 0
        prompt_eval_latency = -1
        token_count = 0
        for chunk in output:
            chunk_text = chunk["response"]
            if chunk["done"]:
                prompt_token_count = chunk.get("prompt_eval_count", 0)
                token_count = chunk.get("eval_count", 0)

            current_time = time.perf_counter()
            latency = current_time - prev_time
            prev_time = current_time
            if prompt_eval_latency == -1:
                prompt_eval_latency = round(latency * 1000, 2)

            yield CompletionResponse(
                text=chunk_text,
                finish_reason="stop" if chunk["done"] else None,
                metrics=CompletionMetrics(
                    prompt_token_count,
                    prompt_eval_latency,
                    token_count,
                    1,
                    round(latency * 1000, 2),
                ),
            )

    def count_tokens(self, text: str) -> int:
        """Get the token count for given text."""
        return 0

    def free(self) -> None:
        if self._llm:
            del self._llm

    @staticmethod
    def validate_config(config: dict) -> bool:
        assert config["name"], "Model name is required for Ollama models."
        return True

    def _get_options(self, data: dict[str, Any]) -> Options:
        """Convert completion settings to ollama options.

        # runtime options
        num_keep: int
        seed: int
        num_predict: int
        top_k: int
        top_p: float
        tfs_z: float
        typical_p: float
        repeat_last_n: int
        temperature: float
        repeat_penalty: float
        presence_penalty: float
        frequency_penalty: float
        mirostat: int
        mirostat_tau: float
        mirostat_eta: float
        penalize_newline: bool
        stop: Sequence[str]
        """
        result = Options()
        for key, val in data.items():
            if key in result:
                result[key] = val
        return result
