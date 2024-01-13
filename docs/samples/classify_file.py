"""A classification sample script."""
#!/usr/bin/env python
import argparse
import glob
import json
import os
from io import StringIO

import frontmatter
from rich.console import Console

from myl.task import Task, create_task, run

console = Console()


def classify_summary_keywords(task: Task, path: str, write: bool) -> None:
    """Classify a text represented by summary and keywords into a category."""
    console.print(f"[bold]> {path}")
    with open(path, "r+") as article:
        post = frontmatter.load(article)
        if "auto_category" in post.metadata:
            console.print("  [yellow]Skipping file since category exists.")
            return
        content = (
            f"```\nsummary: {post['auto_summary']}\n"
            f"keywords: {post['auto_keywords']}\n```"
        )
        instruction = (
            "Classify provided summary and keywords in JSON format: "
            '{"category": ["Technology", "Life", "Philosophy"]}. '
            "Choose only one category and always respond in JSON format. "
            f"Here's the text:\n{content}\n"
        )
        text = StringIO()
        for response in run(task, instruction):
            text.write(response)
        try:
            response = json.loads(text.getvalue())
            post["auto_category"] = response["category"]
            if write:
                frontmatter.dump(post, path, sort_keys=False)
            console.print("---")
            console.print(f'  {post.metadata["auto_summary"]}')
            console.print(f'  {post.metadata["auto_keywords"]}')
            console.print(f'  {post.metadata["category"]}')
            console.print(f'  [green]{post["auto_category"]}[/green]')
            console.print("---")
            console.print(f"[dim]{task.result and task.result.metrics}[/dim]")
        except Exception:
            console.print(text.getvalue())
            console.print_exception()


def main(path: str, write: bool):
    """Classify sample entrypoint."""
    classify_overrides = os.path.join(os.path.dirname(__file__), "classify.yml")
    task, _ = create_task(classify_overrides)

    if os.path.isdir(path):
        for file in glob.iglob(os.path.join(path, "**", "*.md"), recursive=True):
            classify_summary_keywords(task, file, write)
        return

    classify_summary_keywords(task, path, write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a markdown file with existing summary and keywords."
    )
    parser.add_argument("path", type=str, help="Path to markdown file or directory.")
    parser.add_argument(
        "-i", "--inplace", action="store_true", help="Update the file inplace"
    )
    args = parser.parse_args()
    main(args.path, args.inplace)
