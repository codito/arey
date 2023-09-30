# An chat prompt for models.
from typing import List

from myl.models.chat import Message, SenderType
from myl.prompts import Prompt


# See
# <https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGML#prompt-template-vicuna>
class VicunaPrompt(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the user's questions."
        )

    @property
    def stop_words(self) -> List[str]:
        return ["USER:"]

    def get_prompt(self, context: str, query: str) -> str:
        return f"{self.raw_prompt}{context}" f"\nUSER: {query}" f"\nASSISTANT: "

    def format_message(self, message: Message) -> str:
        if message.sender == SenderType.USER:
            return f"\nUSER: {message.text}"
        if message.sender == SenderType.AI:
            return f"\nASSISTANT: {message.text}"
        raise NotImplementedError(f"Unknown sender type: {message.sender}")
