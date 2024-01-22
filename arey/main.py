"""Arey app cli entrypoint."""
#!/usr/bin/env python
import click
import signal
import datetime
from functools import wraps
from typing import Callable, Iterable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from watchfiles import watch

from arey.ai import CompletionMetrics
from arey.model import AreyError
from arey.platform.console import SignalContextManager, get_console
from arey.play import PlayFile


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


def _print_logs(console: Console, verbose: bool, logs: Optional[str]) -> None:
    if not verbose or not logs:
        return
    console.print()
    console.print(logs)
    console.print()


def error_handler(func):
    """Global error handler for Arey."""

    @wraps(func)
    def wrapper(*args, **kwargs):
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

            error_text = Group(
                Markdown(f"ERROR: {e.args[0]}", style="error"),
                Text(),
                Markdown(help_text),
            )
            console.print(error_text)

    return wrapper


@click.command("ask")
@click.argument("instruction", nargs=-1)
@click.option("-o", "--overrides-file", type=click.File())
@click.pass_context
@error_handler
def task(ctx: click.Context, instruction: str, overrides_file: str) -> int:
    """Run an instruction and generate response."""
    from arey.task import create_task, run

    verbose = ctx.obj["VERBOSE"]

    console = get_console()
    console.print()
    console.print("Welcome to arey task!")
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

    _print_logs(console, verbose, task.result and task.result.logs)
    return 0


@click.command("chat")
@click.pass_context
@error_handler
def chat(ctx: click.Context) -> int:
    """Chat with an AI model."""
    from arey.chat import create_chat, get_completion_metrics, stream_response

    verbose = ctx.obj["VERBOSE"]

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


@click.command("play")
@click.argument("file", required=False)
@click.option(
    "--no-watch",
    is_flag=True,
    default=False,
    help="Watch the play file and regenerate response on save.",
)
@click.pass_context
@error_handler
def play(ctx: click.Context, file: str, no_watch: bool) -> int:
    """Watch FILE for model, prompt and generate response on edit.

    If FILE is not provided, a temporary file is created for edit.
    """
    from arey.play import get_play_file, get_play_response, load_play_model

    verbose = ctx.obj["VERBOSE"]

    console = get_console()
    console.print()
    console.print(
        "Welcome to arey play! Edit the play file below in your favorite editor "
        "and I'll generate a response for you. Use `Ctrl+C` to abort play session."
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
                model_metrics = load_play_model(play_file_mod)
                footer = f"✓ Model loaded. {model_metrics.init_latency_ms / 1000:.2f}s."
                console.print(footer, style="message_footer")
        console.print()

        _generate_response(
            console,
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


@click.group()
@click.option("-v", "--verbose", default=False, help="Show verbose logs.")
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """Arey - a simple large language model app."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    pass


main.add_command(task)
main.add_command(chat)
main.add_command(play)

if __name__ == "__main__":
    main()
