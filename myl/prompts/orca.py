# Chat prompt for Orca tuned models

from typing import List

from myl.models.chat import Message, SenderType
from myl.prompts import Prompt


# See
# <https://huggingface.co/stabilityai/StableBeluga-7B#usage>
class OrcaPrompt(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "You are StableBeluga, an AI that follows instructions extremely "
            "well. Help as much as you can. Remember, be safe, and don't do "
            "anything illegal."
        )

    @property
    def stop_words(self) -> List[str]:
        return ["### User:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            f"### System:\n{self.raw_prompt}"
            f"\n\n{context}"
            f"\n\n### User:\n{query}"
            f"\n\n### Assistant:\n"
        )

    def format_message(self, message: Message) -> str:
        if message.sender == SenderType.USER:
            return f"\n\n### User:\n{message.text}"
        if message.sender == SenderType.AI:
            return f"\n\n### Assistant:\n{message.text}"
        raise NotImplementedError(f"Unknown sender type: {message.sender}")
