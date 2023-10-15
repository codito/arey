#!/usr/bin/env python
import argparse
import glob
import json
import os
from io import StringIO

import frontmatter
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from rich.console import Console

from myl.task import Task, create_task, run

console = Console()


def _convert_to_text(markdown: str):
    html_content = MarkdownIt().render(markdown)
    return "".join(BeautifulSoup(html_content, features="lxml").find_all(string=True))


def summarize_markdown_file(task: Task, path: str, write: bool) -> None:
    console.print(f"[bold]> {path}")
    with open(path, "r+") as article:
        post = frontmatter.load(article)
        if "auto_keywords" in post.metadata and "auto_summary" in post.metadata:
            console.print(
                "  [yellow]Skipping file since summary and keywords"
                " already exist.[/yellow]"
            )
            return
        content = _convert_to_text(post.content)
        instruction = (
            "Summarize below text in JSON format: "
            '{"summary": "", "keywords": []}. '
            f"Here's the text:\n```\n{content}\n```\n"
        )
        text = StringIO()
        for response in run(task, instruction):
            text.write(response)
        try:
            summary = json.loads(text.getvalue())
            post["auto_summary"] = summary["summary"]
            post["auto_keywords"] = summary["keywords"]
            if write:
                frontmatter.dump(post, path, sort_keys=False)
            console.print("---")
            console.print(f'  [green]{summary["summary"]}[/green]')
            console.print(f'  [green]{summary["keywords"]}[/green]')
            console.print("---")
            console.print(f"[dim]{task.result and task.result.metrics}[/dim]")
        except Exception:
            console.print(text.getvalue())
            console.print_exception()


def main(path: str, write: bool):
    summarize_overrides = os.path.join(os.path.dirname(__file__), "summarize.yml")
    task, _ = create_task(summarize_overrides)

    if os.path.isdir(path):
        for file in glob.iglob(os.path.join(path, "**", "*.md"), recursive=True):
            summarize_markdown_file(task, file, write)
        return

    summarize_markdown_file(task, path, write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize a markdown file or directory."
    )
    parser.add_argument("path", type=str, help="Path to markdown file or directory.")
    parser.add_argument(
        "-i", "--inplace", action="store_true", help="Update the file inplace"
    )
    args = parser.parse_args()
    main(args.path, args.inplace)
