"""
Prompt templates operates on a simple model of messages.
"""

from abc import ABCMeta, abstractmethod


class PromptTemplate(metaclass=ABCMeta):
    @abstractmethod
    def get_prompt(self, context: str, query: str) -> str:
        raise NotImplementedError
