from typing import List, Tuple

from myl.config import get_config
from myl.models.ai import CompletionMetrics, ModelMetrics
from myl.models.chat import Message
from myl.platform.llama import LlamaBaseModel
from myl.prompts import Prompt

# FIXME dry violation, needs refactoring
config = get_config()
models = config["models"]
chat_model = config["chat"]["model"]
model_path = models[chat_model]["path"]
model: LlamaBaseModel = LlamaBaseModel(model_path=model_path)


# StableBeluga format, or OpenChat format
class SummarizePrompt(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "You're an helpful assistant. Please think step by step and "
            "correctly answer the query."
        )

    @property
    def stop_words(self) -> List[str]:
        return ["### User:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            # f"### System:\n{self.raw_prompt}" # required for StableBeluga
            # f"{context}"
            f"\n\n### User: "
            "Summarize and extract keywords from below text. Your response must "
            'be in this JSON format: {"summary": "", "keywords": []}.\n'
            "Here's the text: \n"
            f"{query}"
            f"\n\n### Assistant: "
        )

    def format_message(self, message: Message) -> str:
        return message.text


# OpenOrcaXOpenChat format
class SummarizePrompt2(Prompt):
    @property
    def raw_prompt(self) -> str:
        return (
            "Summarize and extract keywords from the text provided below. "
            "and readable by a twelve year old. "
            'You must respond in JSON format: {"keywords": [], "summary": ""}.\n'
        )

    @property
    def stop_words(self) -> List[str]:
        return ["### User:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            # f"### System:\n{self.raw_prompt}{context}"
            f"\n\n### User:\n{self.raw_prompt}\n{query}\n"
            f"\n\n### Assistant:\n"
        )

    def format_message(self, message: Message) -> str:
        return message.text


# Alpaca instruction format
class SummarizePrompt3(Prompt):
    instruct_only = True

    @property
    def raw_prompt(self) -> str:
        return (
            "Below is an instruction that describes a task. Write a "
            "response that appropriately completes the request."
        )

    @property
    def stop_words(self) -> List[str]:
        return ["### Instruction:"]

    def get_prompt(self, context: str, query: str) -> str:
        return (
            # f"{self.raw_prompt}{context}"
            f"\n\n### Instruction:\n\n"
            "Summarize and extract keywords from below text. Your response must "
            'be in this JSON format: {"summary": "", "keywords": []}.\n'
            "Summary must be in first person.\n"
            "Here's the text: \n"
            f"{query}"
            f"\n\n### Response:\n"
        )

    def format_message(self, message: Message) -> str:
        return message.text


# FIXME dry violation
def _get_usage(
    model_metrics: ModelMetrics, usage_series: List[CompletionMetrics]
) -> CompletionMetrics:
    response_latency = 0
    response_tokens = 0
    for u in usage_series:
        response_latency += u.latency_ms
        response_tokens += u.completion_tokens
    return CompletionMetrics(
        prompt_tokens=usage_series[0].prompt_tokens,
        completion_tokens=response_tokens,
        total_tokens=usage_series[0].prompt_tokens + response_tokens,
        latency_ms=response_latency,
        time_to_first_token_ms=model_metrics.init_latency_ms
        + usage_series[0].latency_ms,
    )


def summarize(text: str) -> Tuple[str, str, CompletionMetrics]:
    prompt_model = SummarizePrompt3()
    # max_tokens = _get_max_tokens(model, prompt_model, message)
    # context = get_history(model, chat, prompt_model, max_tokens)
    context = ""
    prompt = prompt_model.get_prompt(context, f"```\n{text}\n```")

    usage_series = []
    summary_text = ""
    for chunk in model.complete(prompt, {"stop": prompt_model.stop_words}):
        summary_text += chunk.text
        usage_series.append(chunk.metrics)

    metrics = _get_usage(model.metrics, usage_series)
    return summary_text, prompt, metrics
