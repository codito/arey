# Create a abstract class for chat prompts.
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import List

from myl.models.chat import Message


class Prompt(ABC, metaclass=ABCMeta):
    @abstractproperty
    def raw_prompt(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def stop_words(self) -> List[str]:
        return []

    @abstractmethod
    def get_prompt(self, context: str, query: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_message(self, message: Message) -> str:
        raise NotImplementedError


def get_prompt_template(template_name: str) -> Prompt:
    from myl.prompts.chat import ChatPrompt
    from myl.prompts.instruct import InstructPrompt
    from myl.prompts.openorca import OpenOrcaPrompt
    from myl.prompts.orca import OrcaPrompt

    if template_name == "chat":
        return ChatPrompt()
    if template_name == "instruct":
        return InstructPrompt()
    if template_name == "orca":
        return OrcaPrompt()
    if template_name == "openorca":
        return OpenOrcaPrompt()
