# An instruction prompt for models.
from myl.models.chat import Message, SenderType
from myl.prompts import Prompt


class InstructPrompt(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "Below is an instruction that describes a task. Write a "
            "response that appropriately completes the request."
        )

    @property
    def stop_words(self) -> list[str]:
        return ["### Instruction:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            f"{self.raw_prompt}{context}"
            f"\n\n### Instruction:\n\n{query}"
            f"\n\n### Response:\n\n"
        )

    def format_message(self, message: Message) -> str:
        if message.sender == SenderType.USER:
            return f"\n\n### Instruction:\n\n{message.text}"
        if message.sender == SenderType.AI:
            return f"\n\n### Response:\n\n{message.text}"
        return f"\n\n### System:\n\n{message.text}"
