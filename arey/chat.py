"""Services for the chat command."""
from typing import Iterator, Optional, Tuple

from arey.ai import CompletionMetrics, ModelMetrics, combine_metrics
from arey.config import get_config
from arey.model import Chat, Message, MessageContext, SenderType
from arey.platform.console import capture_stderr
from arey.platform.llama import LlamaBaseModel
from arey.prompt import Prompt, get_prompt

config = get_config()
model_path = config.chat.model.path
prompt_template = config.chat.model.template
model_settings = config.chat.settings
completion_settings = config.chat.profile

model: LlamaBaseModel = LlamaBaseModel(
    model_path=model_path, model_settings=model_settings
)
prompt_model = get_prompt(prompt_template)


def _get_max_tokens(model: LlamaBaseModel, prompt_model: Prompt, text: str) -> int:
    context_size = model.context_size
    prompt_tokens_without_history = model.count_tokens(
        prompt_model.get("chat", {"user_query": text, "chat_history": ""})
    )
    buffer = 200

    return context_size - prompt_tokens_without_history - buffer


def create_chat() -> Tuple[Chat, ModelMetrics]:
    """Create a new chat session."""
    system_prompt = prompt_model.get_message("system", "")
    with capture_stderr() as stderr:
        model.load(system_prompt)
    chat = Chat()
    chat.context.metrics = model.metrics
    chat.context.logs = stderr.getvalue()
    return chat, model.metrics


def get_history(
    model: LlamaBaseModel, chat: Chat, prompt_model: Prompt, max_tokens: int
) -> str:
    """Get the messages for a chat."""
    messages = []
    token_count = 0
    for message in reversed(chat.messages):
        role = (
            "ai"
            if message.sender == SenderType.AI
            else "user"
            if message.sender == SenderType.USER
            else "system"
        )
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
    max_tokens = _get_max_tokens(model, prompt_model, message)
    context = {
        "user_query": message,
        "chat_history": get_history(model, chat, prompt_model, max_tokens),
    }
    prompt = prompt_model.get("chat", context)

    user_msg = Message(message, 0, SenderType.USER, None)
    chat.messages.append(user_msg)

    ai_msg_text = ""
    usage_series = []
    finish_reason = ""
    with capture_stderr() as stderr:
        for chunk in model.complete(prompt, {"stop": prompt_model.stop_words}):
            ai_msg_text += chunk.text
            finish_reason = chunk.finish_reason
            usage_series.append(chunk.metrics)
            yield chunk.text

    msg_context = MessageContext(
        prompt=prompt,
        finish_reason=finish_reason,
        metrics=combine_metrics(usage_series),
        logs=stderr.getvalue(),
    )
    ai_msg = Message(ai_msg_text, 0, SenderType.AI, msg_context)
    chat.messages.append(ai_msg)


def get_completion_metrics(chat: Chat) -> Optional[CompletionMetrics]:
    """Get completion metrics for the chat."""
    msg = next(
        filter(lambda m: m.sender == SenderType.AI, reversed(chat.messages)), None
    )
    return msg.context.metrics if msg and msg.context else None
