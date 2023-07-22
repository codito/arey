#!/usr/bin/env python
import signal
import sys

from myl.platform.console import SignalContextManager, get_console
from myl.services.chat import (create_chat, get_completion_metrics,
                               stream_response)


def chat(args: list[str]) -> int:
    console = get_console()
    console.print(
        ("Welcome to myl chat! How can I help you today?" "\nType 'q' to exit.")
    )
    console.print()

    chat = create_chat()
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
            global stop_completion
            stop_completion = True

        with SignalContextManager(signal.SIGINT, stop_completion_handler):
            for response in stream_response(chat, user_input):
                if stop_completion:
                    break
                console.print(response, end="")

        metrics = get_completion_metrics(chat)
        footer = "◼ Canceled." if stop_completion else "◼ Completed."
        if metrics:
            tokens_per_sec = int(metrics.total_tokens * 1000 / metrics.latency_ms)
            footer += (
                f" {int(metrics.latency_ms / 1000)} seconds."
                f" {tokens_per_sec} tokens/s."
                f" {metrics.total_tokens} tokens."
            )

        console.print("\n")
        console.print(footer, style="message_footer")
        console.print()

    return 0


def main(args: list[str] = sys.argv) -> int:
    commands = {
        "chat": chat,
    }
    if len(args) < 2:
        print("Usage: {} <command>".format(args[0]))
        print("Supported commands: {}".format(", ".join(commands.keys())))
        return 1

    command = args[1]
    return commands[command](args[2:])


if __name__ == "__main__":
    main()
