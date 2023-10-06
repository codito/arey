#!/usr/bin/env python
import signal
import sys
from typing import List

from numpy import who
from rich.console import Group
from rich.live import Live
from rich.progress import Progress
from rich.spinner import Spinner
from rich.text import Text

from myl.chat import create_chat, get_completion_metrics, stream_response
from myl.platform.console import SignalContextManager, get_console


def chat(args: List[str]) -> int:
    console = get_console()
    console.print(
        ("Welcome to myl chat! How can I help you today?" "\nType 'q' to exit.")
    )
    console.print()

    with console.status("Loading model..."):
        chat, model_metrics = create_chat()
        footer = f"✓ Model loaded. {model_metrics.init_latency_ms / 1000:.2f}s."
        console.print(footer, style="message_footer")
        console.print()

    while True:
        # Get input from user
        # Workaround for https://github.com/Textualize/rich/issues/2293
        with console.capture() as capture:
            console.print("> ", style="message_prompt", end="")
        prompt_str = capture.get()
        try:
            user_input = input(prompt_str)
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print("\nBye!")
            break

        if user_input in ["q", "quit"]:
            console.print("Bye!")
            break

        # response = create_response(chat, user_input)
        console.print()
        stop_completion = False

        def stop_completion_handler(signal, frame):
            nonlocal stop_completion
            stop_completion = True

        spinner = Spinner(text="Generating...", name="dots")
        text = Text()
        output = Group(
            text,
            spinner,
        )

        def format_text(text: str) -> str:
            if not text.endswith("\\"):
                return text.encode("utf-8").decode("unicode_escape")
            return text

        with SignalContextManager(signal.SIGINT, stop_completion_handler):
            with Live(console=console, transient=True, refresh_per_second=8) as live:
                live.update(output)
                for response in stream_response(chat, user_input):
                    if stop_completion:
                        break
                    text.append(response)
                    text.plain = format_text(text.plain)
                    live.update(output)

        text = Text(chat.messages[-1].text)
        console.print(text)
        metrics = get_completion_metrics(chat)
        footer = (
            "◼ Canceled."
            if stop_completion
            else f"◼ Completed ({chat.messages[-1].context.finish_reason})."
        )
        if metrics:
            tokens_per_sec = (
                metrics.completion_tokens * 1000 / metrics.completion_latency_ms
            )
            footer += (
                f" {metrics.prompt_eval_latency_ms / 1000:.2f}s to first token."
                f" {metrics.completion_latency_ms / 1000:.2f}s total."
                f" {tokens_per_sec:.2f} tokens/s."
                f" {metrics.completion_tokens} tokens."
                f" {metrics.prompt_tokens} prompt tokens."
            )

        console.print()
        console.print(footer, style="message_footer")
        console.print()

    return 0


def main(args: List[str] = sys.argv) -> int:
    commands = {
        "chat": chat,
        # "ask": ask,
    }
    if len(args) < 2:
        print("Usage: {} <command>".format(args[0]))
        print("Supported commands: {}".format(", ".join(commands.keys())))
        return 1

    command = args[1]
    return commands[command](args[2:])


if __name__ == "__main__":
    main()
