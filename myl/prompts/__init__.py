# Create a abstract class for chat prompts.
import os
from dataclasses import dataclass, field
from functools import lru_cache
from string import Template
from typing import Dict, List, Literal

import yaml

SYSTEM_TOKENS = set(["message_text", "chat_history", "user_query"])


@dataclass
class Prompt:
    name: str
    system_tokens: List[str] = field(default_factory=list)
    custom_tokens: Dict[str, str] = field(default_factory=dict)
    stop_words: List[str] = field(default_factory=list)
    prompts: Dict[str, str] = field(default_factory=dict)  # task: prompt
    message_formats: Dict[str, str] = field(default_factory=dict)  # role: format

    @classmethod
    def create(cls, yml: str) -> "Prompt":
        content = yaml.safe_load(yml) or {}
        name = content.get("name", "")
        if not name:
            raise ValueError("`name` element in the prompt template is " "required.")

        tokens = content.get("tokens", {})
        system_tokens = tokens.get("system", [])
        custom_tokens = tokens.get("custom", {})

        stop_words = content.get("stop_words", [])

        prompts = content.get("prompts", {})
        if not prompts.get("chat") or not prompts.get("task"):
            raise ValueError(
                "`prompts` element in the prompt template is required."
                " It must define `chat` and `task` prompt formats."
            )

        roles = content.get("roles", {})
        if (
            not roles
            or not roles.get("ai")
            or not roles.get("user")
            or not roles.get("system")
        ):
            raise ValueError("`roles` element in the prompt template is required.")
        message_formats = {
            "ai": roles.get("ai").get("message"),
            "user": roles.get("user").get("message"),
            "system": roles.get("system").get("message"),
        }

        return cls(
            name, system_tokens, custom_tokens, stop_words, prompts, message_formats
        )

    def get(self, task: Literal["chat", "task"], context: Dict[str, str]) -> str:
        merged_context = {**context, **self.custom_tokens}
        return Template(self.prompts[task]).substitute(merged_context)

    def get_message(self, role: Literal["ai", "user", "system"], text: str) -> str:
        merged_context = {"message_text": text} | self.custom_tokens
        return Template(self.message_formats[role]).substitute(merged_context)


@lru_cache(maxsize=1)
def _get_oob_prompts() -> Dict[str, Prompt]:
    # get list of yml files in myl/prompts
    result: Dict[str, Prompt] = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for file_path in os.listdir(dir_path):
        if not file_path.endswith(".yml"):
            continue

        try:
            with open(os.path.join(dir_path, file_path), "r") as f:
                prompt = Prompt.create(f.read())
                result[prompt.name] = prompt
        except Exception as err:
            print(f"Error parsing: {file_path}. Error: {err.args[0]}")
            raise

    return result


def get_prompt(template_name: str) -> Prompt:
    oob_prompts = _get_oob_prompts()
    return oob_prompts[template_name]
