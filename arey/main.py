"""Arey app cli entrypoint."""

#!/usr/bin/env python
import datetime
import signal
from collections.abc import Iterable
from functools import wraps
from types import FrameType
from typing import Any, Callable

import click
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text
from watchfiles import watch

from arey.core import AreyError, CompletionMetrics
from arey.platform.console import SignalContextManager, get_console
from arey.play import PlayFile
from arey.task import close


def _generate_response(
    console: Console,
    output_settings: dict[str, str],
    run: Callable[[], Iterable[str]],
    get_metrics: Callable[[], CompletionMetrics | None],
) -> None:
    stop_completion = False

    def stop_completion_handler(_signal: signal.Signals, _frame: FrameType):
        nonlocal stop_completion
        stop_completion = True

    text = Text()
    status = Spinner(
        "dots", text="[message_footer]Generating...", style="message_footer"
    )
    output = Group(status, text)
    with SignalContextManager(signal.SIGINT, stop_completion_handler):
        # Using transient renderable since we replace content with Markdown
        # rendering upon complete. Vertical overflow allows auto scrolling text
        # beyond the terminal height.
        with Live(output, console=console, transient=True, vertical_overflow="visible"):
            for response in run():
                if stop_completion:
                    break
                if len(text) < 1 and status in output.renderables:
                    output.renderables.remove(status)
                text.append(response)

    # Default output in markdown
    plain_text = text.plain.rstrip("\r\n")
    output_format = output_settings.get("format", "markdown")
    if output_format == "plain":
        console.print(plain_text)
    else:
        console.print(Markdown(plain_text))

    metrics = get_metrics()
    footer = "◼ Canceled." if stop_completion else "◼ Completed."
    if metrics:
        tokens_per_sec = (
            metrics.completion_tokens * 1000 / metrics.completion_latency_ms
        )
        tokens = f" {tokens_per_sec:.2f} tokens/s." if tokens_per_sec > 0 else ""
        completion_tokens = (
            f" {metrics.completion_tokens} tokens."
            if metrics.completion_tokens > 0
            else ""
        )
        prompt_tokens = (
            f" {metrics.prompt_tokens} prompt tokens."
            if metrics.prompt_tokens > 0
            else ""
        )
        footer += (
            f" {metrics.prompt_eval_latency_ms / 1000:.2f}s to first token."
            f" {metrics.completion_latency_ms / 1000:.2f}s total."
            f"{tokens}"
            f"{completion_tokens}"
            f"{prompt_tokens}"
        )

    console.print()
    console.print(footer, style="message_footer")
    console.print()


def _print_logs(console: Console, verbose: bool, logs: str | None) -> None:
    if not verbose or not logs:
        return
    console.print()
    console.print(logs)
    console.print()


def error_handler(func: Callable[..., int]):
    """Global error handler for Arey."""

    @wraps(func)
    def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> int:
        try:
            return func(*args, **kwargs)
        except AreyError as e:
            console = get_console()
            help_text: str = (
                "Ouch! This appears to be a bug. Do you mind "
                "connecting with us at "
                "<https://github.com/codito/arey/issues/new>"
            )

            match e.category:
                case "template":
                    help_text = "A template seems misconfigured. Check out the docs."
                case "config":
                    help_text = "Config file seems misconfigured. Check out the docs."
                case _:
                    help_text = "Unknown error."

            error_text = Group(
                Markdown(f"ERROR: {e.args[0]}", style="error"),
                Text(),
                Markdown(help_text),
            )
            console.print(error_text)
            return 1

    return wrapper


def common_options(func: Callable[..., int]):
    """Get common options for arey commands."""

    @click.option(
        "-v", "--verbose", is_flag=True, default=False, help="Show verbose logs."
    )
    @wraps(func)
    def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> int:
        return func(*args, **kwargs)

    return wrapper


@click.group()
def main():
    """Arey - a simple large language model app."""
    pass


@main.command("ask")
@click.argument("instruction", nargs=-1)
@click.option("-o", "--overrides-file", type=click.File())
@error_handler
@common_options
def task(instruction: str, overrides_file: str, verbose: bool) -> int:
    """Run an instruction and generate response."""
    from arey.task import create_task, run

    console = get_console()
    console.print()
    console.print("Welcome to arey ask!")
    console.print()

    with console.status("[message_footer]Loading model..."):
        task, model_metrics = create_task(overrides_file)
        footer = f"✓ Model loaded. {model_metrics.init_latency_ms / 1000:.2f}s."
        console.print(footer, style="message_footer")
        console.print()

    _generate_response(
        console,
        {},
        lambda: run(task, instruction),
        lambda: (task.result and task.result.metrics),
    )

    close(task)
    _print_logs(console, verbose, task.result and task.result.logs)
    return 0


@main.command("chat")
@error_handler
@common_options
def chat(verbose: bool) -> int:
    """Chat with an AI model."""
    import readline  # noqa enable GNU readline capabilities. # pyright: ignore[reportUnusedImport]
    from arey.chat import create_chat, get_completion_metrics, stream_response

    console = get_console()
    console.print(("Welcome to arey chat!\nType 'q' to exit."))
    console.print()

    with console.status("[message_footer]Loading model..."):
        chat, model_metrics = create_chat()
        footer = f"✓ Model loaded. {model_metrics.init_latency_ms / 1000:.2f}s."
        console.print(footer, style="message_footer")
        console.print()

    console.print("How can I help you today?")
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
            {},
            lambda: stream_response(chat, user_input),
            lambda: get_completion_metrics(chat),
        )

        _print_logs(
            console,
            verbose,
            (
                chat.messages[-1].context.logs
                if chat.messages and chat.messages[-1].context
                else ""
            ),
        )

    return 0


@main.command("play")
@click.argument("file", required=False)
@click.option(
    "--no-watch",
    is_flag=True,
    default=False,
    help="Watch the play file and regenerate response on save.",
)
@error_handler
@common_options
def play(file: str, no_watch: bool, verbose: bool) -> int:
    """Watch FILE for model, prompt and generate response on edit.

    If FILE is not provided, a temporary file is created for edit.
    """
    from arey.play import get_play_file, get_play_response, load_play_model

    console = get_console()
    console.print()
    console.print(
        (
            "Welcome to arey play! Edit the play file below in your favorite editor "
            "and I'll generate a response for you. Use `Ctrl+C` to abort play session."
        )
    )
    console.print()

    def run_play_file(play_file_old: PlayFile) -> PlayFile:
        file_path = file or play_file_old.file_path
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print()
        console.rule(current_date)
        play_file_mod = get_play_file(file_path)
        play_file_mod.model = play_file_old.model

        # Compare old playfile with current and reload model if settings have
        # changed
        if (
            play_file_mod.model is None
            or play_file_old.file_path != play_file_mod.file_path
            or play_file_old.model_settings != play_file_mod.model_settings
        ):
            with console.status("[message_footer]Loading model..."):
                try:
                    model_metrics = load_play_model(play_file_mod)
                except AreyError as ae:
                    if ae.category == "config":
                        console.print(ae.message)
                        return play_file_mod
                    raise

                footer = f"✓ Model loaded. {model_metrics.init_latency_ms / 1000:.2f}s."
                console.print(footer, style="message_footer")
        console.print()
        output_settings = play_file_mod.output_settings

        _generate_response(
            console,
            output_settings,
            lambda: get_play_response(play_file_mod),
            lambda: (play_file_mod.result and play_file_mod.result.metrics),
        )

        _print_logs(console, verbose, play_file.result and play_file.result.logs)
        return play_file_mod

    play_file = get_play_file(file)
    if no_watch:
        run_play_file(play_file)
        return 0

    console.print(f"Watching `{play_file.file_path}` for changes...")
    for _ in watch(play_file.file_path):
        play_file = run_play_file(play_file)
        console.print()
        console.print(f"Watching `{play_file.file_path}` for changes...")
    return 0


if __name__ == "__common__":
    main()
