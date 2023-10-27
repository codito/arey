#!/usr/bin/env python
import signal
import sys
from typing import Callable, Iterable, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from myl.ai import CompletionMetrics
from myl.chat import create_chat, get_completion_metrics, stream_response
from myl.platform.console import SignalContextManager, get_console
from myl.task import create_task, run


def _generate_response(
    console: Console,
    run: Callable[[], Iterable[str]],
    get_metrics: Callable[[], Optional[CompletionMetrics]],
) -> None:
    stop_completion = False

    def stop_completion_handler(signal, frame):
        nonlocal stop_completion
        stop_completion = True

    spinner = Spinner(
        text="[message_footer]Generating...", name="dots", style="message_footer"
    )
    text = Text()
    output = Group(
        Padding(text, pad=(0, 0, 2, 0)),
        spinner,
    )

    with SignalContextManager(signal.SIGINT, stop_completion_handler):
        with Live(
            output,
            console=console,
            transient=True,
        ):
            for response in run():
                if stop_completion:
                    break
                text.append(response)

    console.print(Markdown(text.plain))

    console.print()
    metrics = get_metrics()
    footer = "◼ Canceled." if stop_completion else "◼ Completed."
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


def _print_logs(console: Console, logs: Optional[str]) -> None:
    if not logs:
        return
    console.print()
    # console.print(logs)
    console.print()


def task(args: List[str]) -> int:
    if len(args) < 1:
        raise ValueError("No instruction provided")
    instruction = args[0]
    overrides_file = args[1] if len(args) > 1 else None

    console = get_console()
    console.print()
    console.print("Welcome to myl task!")
    console.print()

    with console.status("[message_footer]Loading model..."):
        task, model_metrics = create_task(overrides_file)
        footer = f"✓ Model loaded. {model_metrics.init_latency_ms / 1000:.2f}s."
        console.print(footer, style="message_footer")
        console.print()

    _generate_response(
        console,
        lambda: run(task, instruction),
        lambda: (task.result and task.result.metrics),
    )

    _print_logs(console, task.result and task.result.logs)
    return 0


def chat(args: List[str]) -> int:
    console = get_console()
    console.print(
        ("Welcome to myl chat! How can I help you today?" "\nType 'q' to exit.")
    )
    console.print()

    with console.status("[message_footer]Loading model..."):
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

        console.print()
        _generate_response(
            console,
            lambda: stream_response(chat, user_input),
            lambda: get_completion_metrics(chat),
        )

        _print_logs(
            console,
            (
                chat.messages[-1].context.logs
                if chat.messages and chat.messages[-1].context
                else ""
            ),
        )

    return 0


def main(args: List[str] = sys.argv) -> int:
    commands = {"chat": chat, "task": task}
    if len(args) < 2:
        print("Usage: {} <command>".format(args[0]))
        print("Supported commands: {}".format(", ".join(commands.keys())))
        return 1

    command = args[1]
    return commands[command](args[2:])


if __name__ == "__main__":
    main()
