"""Persona for arey."""
# pyright: basic

# import os
from dataclasses import dataclass

# from functools import lru_cache
#
# from arey.platform.assets import get_asset_dir


@dataclass
class Persona:
    """Persona gives a character to the LLM assistant.

    Arey creates a system prompt based on the various elements in the persona.
    """

    """Name of the persona"""
    name: str

    """Define the persona. E.g., You are an expert philosopher etc."""
    character: str

    """Response instructions are used for creating the final output."""
    response: str

    """Provide a tone for the response."""
    tone: str

    """Examples of input and output pairs for in-context learning."""
    examples: str


# @lru_cache(maxsize=1)
# def _get_oob_personas() -> list[Persona]:
#     # get list of yml files in arey/prompts
#     dir_path = get_asset_dir("prompts")
#     for file_path in os.listdir(dir_path):
#         if not file_path.endswith(".yml"):
#             continue
#
#         try:
#             with open(os.path.join(dir_path, file_path), "r") as f:
#                 prompt = Persona.create(f.read())
#                 result[prompt.name] = prompt
#         except Exception as err:
#             print(f"Error parsing: {file_path}. Error: {err.args[0]}")
#             raise
#
#     return result
#
#
# def get_personas() -> list[Persona]:
#     """Get a list of personas from user's configuration dir."""
#     raise NotImplementedError
