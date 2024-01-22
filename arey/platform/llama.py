"""Llama.cpp based models."""
import dataclasses
import os
import time
import multiprocessing
from typing import Iterator, cast

import llama_cpp

from arey.ai import CompletionMetrics, CompletionModel, CompletionResponse, ModelMetrics
from arey.model import AreyError


@dataclasses.dataclass
class LlamaSettings:
    """Core model settings."""

    n_threads: int = max(multiprocessing.cpu_count() // 2, 1)
    n_ctx: int = 4096
    n_batch: int = 512
    n_gpu_layers: int = 0
    use_mlock: bool = False
    verbose: bool = True


class LlamaBaseModel(CompletionModel):
    """Base local completion model.

    Wraps over the llama-cpp library.
    """

    context_size: int = 4096
    _model_settings: LlamaSettings
    _metrics: ModelMetrics

    def __init__(self, model_path: str, model_settings: dict = {}) -> None:
        """Create an instance of local completion model."""
        self._llm = None
        self._model_path = model_path
        self._model_settings = LlamaSettings(**model_settings)
        self.context_size = self._model_settings.n_ctx

    @property
    def metrics(self) -> ModelMetrics:
        """Get metrics for model initialization."""
        return self._metrics

    def _get_model(self):
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
                **dataclasses.asdict(self._model_settings),
            )
            # self._llm.set_cache(llama_cpp.LlamaCache(2 << 33))

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._metrics = ModelMetrics(init_latency_ms=latency_ms)
        return self._llm

    def load(self, text: str):
        """Load a model into memory."""
        model = self._get_model()
        model.eval(model.tokenize(text.encode("utf-8")))

    def complete(self, text: str, settings: dict = {}) -> Iterator[CompletionResponse]:
        """Get a completion for the given text and settings."""
        prev_time = time.perf_counter()
        model = self._get_model()
        completion_settings = {
            "prompt": text,
            "max_tokens": -1,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.1,
            "repeat_penalty": 1.176,
            "echo": False,
        } | settings
        output = cast(
            Iterator[llama_cpp.CompletionChunk],
            model.create_completion(
                **completion_settings,
                stream=True,
            ),
        )

        prompt_token_count = self.count_tokens(text)
        prompt_eval_latency = -1
        for chunk in output:
            chunk_text = chunk["choices"][0]["text"]

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

    def count_tokens(self, text: str) -> int:
        """Get the token count for given text."""
        model = self._get_model()
        return len(model.tokenize(text.encode("utf-8")))
