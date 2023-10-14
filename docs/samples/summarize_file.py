#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from io import StringIO
from typing import List, Tuple

import frontmatter
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from myl.task import Task, create_task, run


@dataclass
class SummaryResponse:
    summary: str
    keywords: List[str]


def _convert_to_text(markdown: str):
    html_content = MarkdownIt().render(markdown)
    return "".join(BeautifulSoup(html_content, features="lxml").find_all(string=True))


def summarize_markdown_file(task: Task, path: str, write: bool) -> None:
    with open(path, "r+") as article:
        post = frontmatter.load(article)
        if "auto_keyword" in post.metadata and "auto_summary" in post.metadata:
            print(f"[SKIP] {path}")
            return
        print(path)
        content = _convert_to_text(post.content)
        instruction = (
            "Summarize below text in JSON format: "
            '{"summary": "", "keywords": []}. '
            f"Here's the text:\n```\n{content}\n```\n"
        )
        text = StringIO()
        for response in run(task, instruction):
            text.write(response)
        summary = json.loads(text.getvalue())
        post["auto_summary"] = summary["summary"]
        post["auto_keywords"] = summary["keywords"]
        if write:
            frontmatter.dump(post, path)
        print("---")
        print(f'  {summary["summary"]}')
        print(f'  {summary["keywords"]}')
        print("---")
        print(task.result and task.result.metrics)


def main(path: str, write: bool):
    summarize_overrides = os.path.join(os.path.dirname(__file__), "summarize.yml")
    task, _ = create_task(summarize_overrides)

    if os.path.isdir(path):
        for file in glob.iglob(os.path.join(path, "**", "*.md"), recursive=True):
            summarize_markdown_file(task, file, write)
        return

    summarize_markdown_file(task, path, write)


if __name__ == "__main__":
    # create argument parser with 1 positional and 1 boolean switch
    parser = argparse.ArgumentParser(
        description="Summarize a markdown file or directory."
    )
    parser.add_argument("path", type=str, help="Path to markdown file or directory.")
    parser.add_argument(
        "-i", "--inplace", action="store_true", help="Update the file inplace"
    )
    args = parser.parse_args()
    main(args.path, args.inplace)
