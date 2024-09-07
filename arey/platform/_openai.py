"""OpenAI API based models."""

import dataclasses
import time
from typing import Iterator, Optional
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from arey.ai import CompletionMetrics, CompletionModel, CompletionResponse, ModelMetrics
from arey.ai import ChatMessage
from arey.error import AreyError


@dataclasses.dataclass
class OpenAISettings:
    """Core model settings."""

    base_url: str
    api_key: Optional[str] = "DUMMY KEY"


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
    def context_size(self) -> int:
        return 0

    @property
    def metrics(self) -> ModelMetrics:
        """Get metrics for model initialization."""
        return ModelMetrics(init_latency_ms=0)

    def load(self, text: str):
        """Load a model into memory."""
        # No-op since these are remote models.
        pass

    def complete(
        self, messages: list[ChatMessage], settings: dict = {}
    ) -> Iterator[CompletionResponse]:
        """Get a completion for the given text and settings."""
        assert self._client

        prev_time = time.perf_counter()
        completion_settings = {
            "temperature": 0.7,
        } | settings

        # FIXME invalid code
        formatted_messages = [self._get_message_from_chat(m) for m in messages]
        output = self._client.chat.completions.create(
            model=self._model_name,
            messages=formatted_messages,
            stream=True,
            temperature=completion_settings["temperature"],
        )
        prompt_token_count = sum(self.count_tokens(m.text) for m in messages)
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
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model_name)
            return len(enc.encode(text))
        except KeyError:
            # Allow OpenAI compatible server endpoints
            pass
        return 0

    def free(self) -> None:
        if self._client:
            del self._client

    @staticmethod
    def validate_config(config: dict) -> bool:
        assert config["name"], "Model name is required for OpenAI models."
        return True

    def _get_message_from_text(self, text: str) -> ChatCompletionMessageParam:
        return ChatCompletionUserMessageParam({"role": "user", "content": text})

    def _get_message_from_chat(
        self, message: ChatMessage
    ) -> ChatCompletionMessageParam:
        if message.sender.role() == "system":
            return ChatCompletionSystemMessageParam(
                {"role": "system", "content": message.text}
            )
        if message.sender.role() == "user":
            return ChatCompletionUserMessageParam(
                {"role": "user", "content": message.text}
            )
        if message.sender.role() == "assistant":
            return ChatCompletionAssistantMessageParam(
                {"role": "assistant", "content": message.text}
            )
        raise AreyError("system", f"Unknown message role: {message.sender.role()}")
