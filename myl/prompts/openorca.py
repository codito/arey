# An instruction prompt for models.
from typing import List

from myl.models.chat import Message, SenderType
from myl.prompts import Prompt


# See
# <https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B#prompt-template>
class OpenOrcaPrompt(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "You are a helpful assistant. Please think step by step and "
            "answer the query."
        )

    @property
    def stop_words(self) -> List[str]:
        return ["User:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            f"{self.raw_prompt}{context}"
            f"\nUser: {query}<|end_of_turn|>"
            f"\nAssistant:\n"
        )

    def format_message(self, message: Message) -> str:
        if message.sender == SenderType.USER:
            return f"\nUser: {message.text}<|end_of_turn|>"
        if message.sender == SenderType.AI:
            return f"\nAssistant: {message.text}<|end_of_turn|>"
        return f"\nSystem: {message.text}<|end_of_turn|>"
