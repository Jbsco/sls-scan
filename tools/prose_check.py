#!/usr/bin/env python3
"""Prose gate for the markdown pages: no list item starts with an article, no paragraph or
list item runs past 100 words, and "the" stays under 8 per 100 words in each file. Usage:
prose_check.py <file.md>..."""
import re
import sys

ARTICLE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(?:\*\*)?(the|a|an)\b", re.IGNORECASE)
ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
WORD = re.compile(r"[A-Za-z']+")
LIMIT_WORDS = 100
LIMIT_THE = 8.0


def check(path):
    problems = []
    lines = open(path, encoding="utf-8").read().split("\n")
    para, start = [], 0

    def flush():
        if para:
            n = len(" ".join(para).split())
            if n > LIMIT_WORDS:
                problems.append(f"{path}:{start}: paragraph of {n} words (limit {LIMIT_WORDS})")

    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or line.startswith("    ") or s[0] in "|#<":
            flush()
            para = []
            continue
        if ARTICLE.match(line):
            problems.append(f"{path}:{i}: list item starts with an article: {s[:60]}")
        if ITEM.match(line):
            flush()
            para, start = [s], i
        else:
            if not para:
                start = i
            para.append(s)
    flush()
    words = WORD.findall(open(path, encoding="utf-8").read())
    the = sum(1 for w in words if w.lower() == "the")
    density = 100.0 * the / max(len(words), 1)
    if density > LIMIT_THE:
        problems.append(f"{path}: {density:.1f} \"the\" per 100 words (limit {LIMIT_THE:.0f})")
    return problems


def main():
    problems = [p for path in sys.argv[1:] for p in check(path)]
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
