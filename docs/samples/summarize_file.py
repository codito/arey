#!/usr/bin/env python
import json
import os
import sys
from typing import List, Tuple

import frontmatter
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from myl.services.summarize import summarize


def _convert_to_text(markdown: str):
    html_content = MarkdownIt().render(markdown)
    return "".join(BeautifulSoup(html_content, features="lxml").find_all(string=True))


def summarize_markdown_file(path: str) -> str:
    with open(path, "r+") as article:
        post = frontmatter.load(article)
        if "keyword" in post.metadata and "summary" in post.metadata:
            print(f"[SKIP] {path}")
        content = _convert_to_text(post.content)
        summary, prompt, metrics = summarize(content)
        # print("---")
        # print(prompt)
        print("---")
        print(metrics)
        return summary


if __name__ == "__main__":
    show_help = len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help"
    if show_help:
        print("Usage: {} <path to markdown file>".format(sys.argv[0]))
        sys.exit(1)
    summary = summarize_markdown_file(sys.argv[1])
    print(summary)
