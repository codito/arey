"""Myl app cli entrypoint."""
#!/usr/bin/env python
import click
import signal
import datetime
from typing import Callable, Iterable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from watchfiles import watch

from aye.ai import CompletionMetrics
from aye.chat import create_chat, get_completion_metrics, stream_response
from aye.platform.console import SignalContextManager, get_console
from aye.task import create_task, run
from aye.play import get_play_file, get_play_response, load_play_model


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
    console.print(logs)
    console.print()


@click.command("play")
@click.argument("file", required=False)
@click.option(
    "--no-watch",
    is_flag=True,
    default=False,
    help="Watch the play file and regenerate response on save.",
)
def play(file: str, no_watch: bool) -> int:
    """Watch FILE for model, prompt and generate response on edit.

    If FILE is not provided, a temporary file is created for edit.
    """
    console = get_console()
    console.print()
    console.print("Welcome to aye play!")
    console.print()

    play_file = get_play_file(file)

    def run_play_file():
        file_path = file or play_file.file_path
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print()
        console.rule(current_date)
        play_file_mod = get_play_file(file_path)
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

        _print_logs(console, play_file.result and play_file.result.logs)

    if no_watch:
        run_play_file()
        return 0

    console.print(f"Watching `{play_file.file_path}` for changes...")
    for _ in watch(play_file.file_path):
        run_play_file()
        console.print()
        console.print(f"Watching `{play_file.file_path}` for changes...")
    return 0


@click.command("task")
@click.argument("instruction", nargs=-1)
@click.option("-o", "--overrides-file", type=click.File())
def task(instruction: str, overrides_file: str) -> int:
    """Run an instruction and generate response."""
    console = get_console()
    console.print()
    console.print("Welcome to aye task!")
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


@click.command("chat")
def chat() -> int:
    """Chat with an AI model."""
    console = get_console()
    console.print(("Welcome to aye chat!\nType 'q' to exit."))
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
            (
                chat.messages[-1].context.logs
                if chat.messages and chat.messages[-1].context
                else ""
            ),
        )

    return 0


@click.group()
def main():
    """Myl - a simple large language model app."""
    pass


main.add_command(chat)
main.add_command(task)
main.add_command(play)

if __name__ == "__main__":
    main()
