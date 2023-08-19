# Llama based models
import os
import time
from dataclasses import dataclass
from typing import Iterator, cast

import llama_cpp
from wurlitzer import pipes

from myl.models.ai import (CompletionMetrics, CompletionModel,
                           CompletionResponse, ModelMetrics)


@dataclass
class LlamaSettings:
    pass


class LlamaBaseModel(CompletionModel):
    context_size = 4096
    _metrics: ModelMetrics

    def __init__(self, model_path: str):
        self._llm = None
        self._model_path = model_path

    @property
    def metrics(self) -> ModelMetrics:
        return self._metrics

    def _get_model(self):
        model_path = os.path.join(os.path.expanduser(self._model_path))
        if not self._llm:
            start_time = time.perf_counter()
            with pipes():
                self._llm = llama_cpp.Llama(
                    model_path=model_path,
                    n_batch=256,
                    n_ctx=self.context_size,
                    n_threads=32,
                    verbose=False,
                )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._metrics = ModelMetrics(init_latency_ms=latency_ms)
        return self._llm

    def complete(self, text: str, settings: dict = {}) -> Iterator[CompletionResponse]:
        model = self._get_model()
        output = cast(
            Iterator[llama_cpp.CompletionChunk],
            model.create_completion(
                prompt=text,
                max_tokens=256,
                temperature=0.7,
                top_k=40,
                top_p=0.1,
                repeat_penalty=1.176,
                echo=False,
                **settings,
                stream=True,
            ),
        )

        prompt_token_count = self.count_tokens(text)
        prev_time = time.perf_counter()
        time_to_first_token = -1
        for chunk in output:
            current_time = time.perf_counter()
            latency = current_time - prev_time
            prev_time = current_time
            if time_to_first_token == -1:
                time_to_first_token = latency

            chunk_text = chunk["choices"][0]["text"]
            token_count = self.count_tokens(chunk_text)
            yield CompletionResponse(
                text=chunk_text,
                metrics=CompletionMetrics(
                    prompt_token_count,
                    token_count,
                    0,
                    round(latency * 1000, 2),
                    time_to_first_token,
                ),
            )

    def count_tokens(self, text: str) -> int:
        model = self._get_model()
        return len(model.tokenize(text.encode("utf-8")))
