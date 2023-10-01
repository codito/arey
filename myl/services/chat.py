# Services for the chat command


from typing import Iterator, List, Optional

from myl.config import get_config
from myl.models.ai import CompletionMetrics, ModelMetrics
from myl.models.chat import Chat, Message, MessageContext, SenderType
from myl.platform.llama import LlamaBaseModel
from myl.prompts import Prompt
from myl.prompts.chat import ChatPrompt
from myl.prompts.instruct import InstructPrompt
from myl.prompts.openorca import OpenOrcaPrompt
from myl.prompts.orca import OrcaPrompt

config = get_config()
model_path = config.chat.model.path
model_settings = config.chat.settings
completion_settings = config.chat.profile

model: LlamaBaseModel = LlamaBaseModel(
    model_path=model_path, model_settings=model_settings
)


def _count_token(model: LlamaBaseModel, text: str) -> int:
    return model.count_tokens(text)


def _get_max_tokens(model: LlamaBaseModel, prompt_model: Prompt, text: str) -> int:
    context_size = model.context_size
    prompt_raw_tokens = _count_token(model, prompt_model.raw_prompt)
    query_tokens = _count_token(model, text)
    buffer = 200

    return context_size - prompt_raw_tokens - query_tokens - buffer


def _get_usage(
    model_metrics: ModelMetrics, usage_series: List[CompletionMetrics]
) -> CompletionMetrics:
    response_latency = 0
    response_tokens = 0
    for u in usage_series:
        response_latency += u.latency_ms
        response_tokens += u.completion_tokens
    return CompletionMetrics(
        prompt_tokens=usage_series[0].prompt_tokens,
        completion_tokens=response_tokens,
        total_tokens=usage_series[0].prompt_tokens + response_tokens,
        latency_ms=response_latency,
        time_to_first_token_ms=model_metrics.init_latency_ms
        + usage_series[0].latency_ms,
    )


def create_chat():
    return Chat()


def get_history(
    model: LlamaBaseModel, chat: Chat, prompt_model: Prompt, max_tokens: int
) -> str:
    messages = []
    token_count = 0
    for message in reversed(chat.messages):
        formatted_message = prompt_model.format_message(message)
        message_tokens = _count_token(model, formatted_message)
        messages.insert(0, formatted_message)

        token_count += message_tokens
        if message.sender == SenderType.USER and token_count >= max_tokens:
            break

    return "".join(messages)


def create_response(chat: Chat, message: str) -> str:
    response = ""
    for chunk in stream_response(chat, message):
        response += chunk

    return response


def stream_response(chat: Chat, message: str) -> Iterator[str]:
    prompt_model = InstructPrompt()
    max_tokens = _get_max_tokens(model, prompt_model, message)
    context = get_history(model, chat, prompt_model, max_tokens)
    prompt = prompt_model.get_prompt(context, message)

    user_msg = Message(message, 0, SenderType.USER, None)
    chat.messages.append(user_msg)

    ai_msg_text = ""
    usage_series = []
    for chunk in model.complete(prompt, {"stop": prompt_model.stop_words}):
        ai_msg_text += chunk.text
        usage_series.append(chunk.metrics)
        yield chunk.text

    msg_context = MessageContext(
        prompt=prompt, metrics=_get_usage(model.metrics, usage_series)
    )
    ai_msg = Message(ai_msg_text, 0, SenderType.AI, msg_context)
    chat.messages.append(ai_msg)


def get_completion_metrics(chat: Chat) -> Optional[CompletionMetrics]:
    msg = next(
        filter(lambda m: m.sender == SenderType.AI, reversed(chat.messages)), None
    )
    return msg.context.metrics if msg and msg.context else None
