"""OpenAI API based models."""
import dataclasses
import time
from functools import reduce
from typing import Iterator
from openai import OpenAI
from arey.ai import CompletionMetrics, CompletionModel, CompletionResponse, ModelMetrics
from arey.model import Message


@dataclasses.dataclass
class OpenAISettings:
    """Core model settings."""

    base_url: str
    api_key: str


class OpenAIBaseModel(CompletionModel):
    """Base OpenAI completion model."""

    _client: OpenAI
    _model_settings: OpenAISettings

    def __init__(self, model_name: str, model_settings: dict = {}) -> None:
        """Create an instance of openai completion model."""
        self._model_name = model_name
        self._model_settings = OpenAISettings(**model_settings)
        self._client = OpenAI(**dataclasses.asdict(self._model_settings))

    @property
    def metrics(self) -> ModelMetrics:
        """Get metrics for model initialization."""
        return ModelMetrics(init_latency_ms=0)

    def load(self, text: str):
        """Load a model into memory."""
        # No-op since these are remote models.
        pass

    def chat_complete(
        self, messages: list[Message], settings: dict = {}
    ) -> Iterator[CompletionResponse]:
        """Get a completion for the given text and settings."""
        assert self._client

        prev_time = time.perf_counter()
        # completion_settings = {
        #     "prompt": text,
        #     "max_tokens": -1,
        #     "temperature": 0.7,
        #     "top_k": 40,
        #     "top_p": 0.1,
        #     "repeat_penalty": 1.176,
        #     "echo": False,
        # } | settings

        # FIXME invalid code
        output = self._client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            stream=True,
        )
        prompt_token_count = reduce(
            lambda val, text: val + self.count_tokens(text),
            [m.text for m in messages],
            0,
        )
        prompt_eval_latency = -1
        for chunk in output:
            chunk_text = chunk.choices[0].delta.content or ""
            finish_reason = chunk.choices[0].finish_reason

            current_time = time.perf_counter()
            latency = current_time - prev_time
            prev_time = current_time
            if prompt_eval_latency == -1:
                prompt_eval_latency = round(latency * 1000, 2)

            token_count = self.count_tokens(chunk_text)
            yield CompletionResponse(
                text=chunk_text,
                finish_reason=finish_reason,
                metrics=CompletionMetrics(
                    prompt_token_count,
                    prompt_eval_latency,
                    token_count,
                    1,
                    round(latency * 1000, 2),
                ),
            )
        pass

    def count_tokens(self, text: str) -> int:
        """Get the token count for given text."""
        import tiktoken

        enc = tiktoken.encoding_for_model(self._model_name)
        return len(enc.encode(text))
