"""Services for the chat command."""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple

from arey.ai import (
    ChatMessage,
    CompletionMetrics,
    CompletionModel,
    ModelMetrics,
    SenderType,
    combine_metrics,
)
from arey.config import get_config
from arey.platform.console import capture_stderr
from arey.platform.llm import get_completion_llm
from arey.prompt import Prompt, get_prompt

config = get_config()
prompt_template = config.chat.model.template
model_settings = config.chat.settings
completion_settings = config.chat.profile

model: CompletionModel = get_completion_llm(
    config.chat.model.asdict(), settings=model_settings
)
prompt_model = get_prompt(prompt_template) if prompt_template else None


@dataclass
class MessageContext:
    """Context associated with a single chat message."""

    prompt: str
    finish_reason: Optional[str]
    metrics: CompletionMetrics
    logs: str = ""


@dataclass(kw_only=True)
class Message(ChatMessage):
    """A chat message with context."""

    timestamp: int  # unix timestamp
    context: Optional[MessageContext]


@dataclass
class ChatContext:
    """Context associated with a chat."""

    metrics: Optional[ModelMetrics] = None
    logs: str = ""


@dataclass
class Chat:
    """A chat conversation between human and AI model."""

    messages: List[Message] = field(default_factory=list)
    context: ChatContext = field(default_factory=ChatContext)


def _get_max_tokens(model: CompletionModel, prompt_model: Prompt, text: str) -> int:
    context_size = model.context_size
    prompt_tokens_without_history = model.count_tokens(
        prompt_model.get("chat", {"user_query": text, "chat_history": ""})
    )
    buffer = 200

    return context_size - prompt_tokens_without_history - buffer


def create_chat() -> Tuple[Chat, ModelMetrics]:
    """Create a new chat session."""
    system_prompt = prompt_model.get_message("system", "") if prompt_model else ""
    with capture_stderr() as stderr:
        model.load(system_prompt)
    chat = Chat()
    chat.context.metrics = model.metrics
    chat.context.logs = stderr.getvalue()
    return chat, model.metrics


def get_history(
    model: CompletionModel, chat: Chat, prompt_model: Prompt, max_tokens: int
) -> str:
    """Get the messages for a chat."""
    messages = []
    token_count = 0
    for message in reversed(chat.messages):
        role = message.sender.role()
        formatted_message = prompt_model.get_message(role, message.text)
        message_tokens = model.count_tokens(formatted_message)
        messages.insert(0, formatted_message)

        token_count += message_tokens
        if message.sender == SenderType.USER and token_count >= max_tokens:
            break

    return "".join(messages)


def create_response(chat: Chat, message: str) -> str:
    """Create a chat response."""
    response = ""
    for chunk in stream_response(chat, message):
        response += chunk

    return response


def stream_response(chat: Chat, message: str) -> Iterator[str]:
    """Stream a chat response."""
    # max_tokens = _get_max_tokens(model, prompt_model, message)
    # context = {
    #     "user_query": message,
    #     "chat_history": get_history(model, chat, prompt_model, max_tokens),
    # }
    # prompt = prompt_model.get("chat", context)

    user_msg = Message(text=message, sender=SenderType.USER, timestamp=0, context=None)
    chat.messages.append(user_msg)

    ai_msg_text = ""
    usage_series = []
    finish_reason = ""
    with capture_stderr() as stderr:
        # for chunk in model.complete(chat.messages, {"stop": prompt_model.stop_words}):
        for chunk in model.complete(chat.messages, {}):
            ai_msg_text += chunk.text
            finish_reason = chunk.finish_reason
            usage_series.append(chunk.metrics)
            yield chunk.text

    msg_context = MessageContext(
        prompt="",
        finish_reason=finish_reason,
        metrics=combine_metrics(usage_series),
        logs=stderr.getvalue(),
    )
    ai_msg = Message(
        text=ai_msg_text, timestamp=0, sender=SenderType.ASSISTANT, context=msg_context
    )
    chat.messages.append(ai_msg)


def get_completion_metrics(chat: Chat) -> Optional[CompletionMetrics]:
    """Get completion metrics for the chat."""
    msg = next(
        filter(lambda m: m.sender == SenderType.ASSISTANT, reversed(chat.messages)),
        None,
    )
    return msg.context.metrics if msg and msg.context else None
