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
