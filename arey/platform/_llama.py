"""Llama.cpp based models."""
# pyright: strict, reportUnknownVariableType=false, reportUnknownMemberType=false

import multiprocessing
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, cast, override

import llama_cpp
from pydantic import BaseModel, model_validator

from arey.core import (
    AreyError,
    ChatMessage,
    CompletionMetrics,
    CompletionModel,
    CompletionResponse,
    ModelConfig,
    ModelMetrics,
)


class LlamaSettings(BaseModel):
    """Core model settings."""

    path: Path
    """Path of the GGUF file."""

    n_threads: int = max(multiprocessing.cpu_count() // 2, 1)
    n_ctx: int = 4096
    n_batch: int = 512
    n_gpu_layers: int = 0
    use_mlock: bool = False
    verbose: bool = True

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Ensure path is valid."""
        if self.path:
            self.path = self.path.expanduser()
            if self.path.exists():
                return self

        raise AreyError("config", f"Model path '{self.path}' for is invalid.")


class LlamaBaseModel(CompletionModel):
    """Base local completion model.

    Wraps over the llama-cpp library.
    """

    _model_path: Path
    _model_settings: LlamaSettings
    _metrics: ModelMetrics

    def __init__(self, model_config: ModelConfig) -> None:
        """Create an instance of local completion model."""
        # FIXME
        # if llama_cpp is None:
        #     raise AreyError(
        #         "config",
        #         f"Invalid configuration. Need `llama-cpp-python` for {model_path}",
        #     )

        self._llm = None
        self._model_settings = LlamaSettings(**model_config.settings)  # pyright: ignore[reportAny]
        self._model_path = self._model_settings.path

    @property
    @override
    def context_size(self) -> int:
        """Get context size for the model."""
        assert self._llm, "Please load the model first with load()."
        return self._llm.n_ctx()

    @property
    @override
    def metrics(self) -> ModelMetrics:
        """Get metrics for model initialization."""
        return self._metrics

    def _get_model(self) -> llama_cpp.Llama:
        model_path = os.path.join(os.path.expanduser(self._model_path))
        if not os.path.exists(model_path):
            raise AreyError(
                "system",
                f"Invalid model path: {model_path}.",
            )
        if not self._llm:
            start_time = time.perf_counter()
            self._llm = llama_cpp.Llama(
                model_path=model_path,
                **self._model_settings.model_dump(),  # pyright: ignore[reportAny]
            )
            # self._llm.set_cache(llama_cpp.LlamaCache(2 << 33))

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._metrics = ModelMetrics(init_latency_ms=latency_ms)
        return self._llm

    def _get_message_from_chat(
        self, message: ChatMessage
    ) -> llama_cpp.ChatCompletionRequestMessage:
        if message.sender.role() == "system":
            return llama_cpp.ChatCompletionRequestSystemMessage(
                {"role": "system", "content": message.text}
            )
        if message.sender.role() == "user":
            return llama_cpp.ChatCompletionRequestUserMessage(
                {"role": "user", "content": message.text}
            )
        if message.sender.role() == "assistant":
            return llama_cpp.ChatCompletionRequestAssistantMessage(
                {"role": "assistant", "content": message.text}
            )
        raise AreyError("system", f"Unknown message role: {message.sender.role()}")

    @override
    def load(self, text: str):
        """Load a model into memory."""
        model = self._get_model()
        model.eval(model.tokenize(text.encode("utf-8")))

    @override
    def complete(
        self, messages: list[ChatMessage], settings: dict[str, Any] | None = None
    ) -> Iterator[CompletionResponse]:
        """Get a completion for the given text and settings."""
        if settings is None:
            settings = {}

        prev_time = time.perf_counter()
        model = self._get_model()
        completion_settings: dict[str, Any] = {
            "max_tokens": -1,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.1,
            "repeat_penalty": 1.176,
        } | settings

        formatted_messages = [self._get_message_from_chat(m) for m in messages]
        output = cast(
            Iterator[llama_cpp.CreateChatCompletionStreamResponse],
            model.create_chat_completion(
                messages=formatted_messages,
                **completion_settings,  # pyright: ignore[reportAny]
                stream=True,
            ),
        )
        prompt_token_count = sum(self.count_tokens(m.text) for m in messages)
        prompt_eval_latency = -1
        for chunk in output:
            chunk_text: str = cast(str, chunk["choices"][0]["delta"].get("content", ""))

            current_time = time.perf_counter()
            latency = current_time - prev_time
            prev_time = current_time
            if prompt_eval_latency == -1:
                prompt_eval_latency = round(latency * 1000, 2)

            token_count = self.count_tokens(chunk_text)
            yield CompletionResponse(
                text=chunk_text,
                finish_reason=chunk["choices"][0]["finish_reason"],
                metrics=CompletionMetrics(
                    prompt_token_count,
                    prompt_eval_latency,
                    token_count,
                    1,
                    round(latency * 1000, 2),
                ),
            )

    @override
    def count_tokens(self, text: str) -> int:
        """Get the token count for given text."""
        model = self._get_model()
        return len(model.tokenize(text.encode("utf-8")))

    @override
    def free(self) -> None:
        if self._llm:
            del self._llm
