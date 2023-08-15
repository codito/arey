#!/usr/bin/env python
import sys
from typing import List

from myl.tasks.chat import chat
from myl.tasks.summarize import summarize


def main(args: List[str] = sys.argv) -> int:
    commands = {
        "chat": chat,
        "summarize": summarize,
    }
    if len(args) < 2:
        print("Usage: {} <command>".format(args[0]))
        print("Supported commands: {}".format(", ".join(commands.keys())))
        return 1

    command = args[1]
    return commands[command](args[2:])


if __name__ == "__main__":
    main()
