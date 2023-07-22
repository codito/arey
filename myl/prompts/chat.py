# Base prompt for chat.
from myl.models.chat import Message, SenderType
from myl.prompts import Prompt


class ChatPrompt(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "A chat between a curious human and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the human's questions."
        )

    @property
    def stop_words(self) -> list[str]:
        return ["### Human:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            f"{self.raw_prompt}{context}"
            f"\n\n### Human:\n{query}"
            f"\n\n### Assistant:\n"
        )

    def format_message(self, message: Message) -> str:
        if message.sender == SenderType.USER:
            return f"\n\n### Human:\n{message.text}"
        if message.sender == SenderType.AI:
            return f"\n\n### Assistant:\n{message.text}"
        return f"\n\n### System:\n{message.text}"
