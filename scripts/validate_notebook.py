#!/usr/bin/env python3
"""Validate explicitly selected tutorial notebooks before contribution."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote, urlparse


REQUIRED_SECTIONS = (
    "Overview",
    "Detailed Explanation",
    "Required Packages",
    "Implementation",
    "Usage Example",
    "Comparison",
    "Additional Considerations",
    "References",
)
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


class ImageSourceParser(HTMLParser):
    """Collect image sources from notebook HTML fragments."""

    def __init__(self) -> None:
        super().__init__()
        self.sources: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag.lower() != "img":
            return
        source = dict(attrs).get("src")
        if source:
            self.sources.append(source)


def source_text(cell: Dict[str, Any]) -> str:
    """Return a cell's source whether nbformat stored it as a string or list."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def normalize_heading(value: str) -> str:
    """Normalize display punctuation so template headings compare by words."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def has_valid_source(cell: Dict[str, Any]) -> bool:
    """Return whether a cell source follows the nbformat string shape."""
    source = cell.get("source", "")
    return isinstance(source, str) or (
        isinstance(source, list) and all(isinstance(part, str) for part in source)
    )


def validate_notebook(path: Path) -> List[Finding]:
    """Return all contribution-readiness findings for one notebook path."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [Finding("NB001", f"cannot read notebook JSON: {exc}")]

    if (
        not isinstance(notebook, dict)
        or notebook.get("nbformat") != 4
        or not isinstance(notebook.get("cells"), list)
        or not all(
            isinstance(cell, dict) and has_valid_source(cell)
            for cell in notebook["cells"]
        )
    ):
        return [Finding("NB001", "expected an nbformat 4 notebook with a cells list")]

    findings: List[Finding] = []
    cells = notebook["cells"]

    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            findings.append(
                Finding("NB002", f"cell {index} has outputs or an execution count")
            )
        previous = cells[index - 2] if index > 1 else {}
        if (
            previous.get("cell_type") != "markdown"
            or HEADING_RE.search(source_text(previous)) is None
        ):
            findings.append(
                Finding("NB003", f"cell {index} needs a preceding markdown description")
            )

    markdown_text = "\n".join(
        source_text(cell) for cell in cells if cell.get("cell_type") == "markdown"
    )
    headings = {normalize_heading(match) for match in HEADING_RE.findall(markdown_text)}
    for section in REQUIRED_SECTIONS:
        normalized = normalize_heading(section)
        if normalized not in headings:
            findings.append(Finding("NB004", f"missing required section: {section}"))

    html_parser = ImageSourceParser()
    html_parser.feed(markdown_text)
    image_targets = [*IMAGE_RE.findall(markdown_text), *html_parser.sources]
    for target in image_targets:
        candidate = target.split(maxsplit=1)[0]
        try:
            parsed = urlparse(candidate)
        except ValueError as exc:
            findings.append(Finding("NB005", f"invalid image target: {target} ({exc})"))
            continue
        if parsed.scheme or parsed.netloc or candidate.startswith(("#", "data:")):
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError:
            findings.append(
                Finding("NB005", f"local image is outside repository root: {target}")
            )
            continue
        if not resolved.is_file():
            findings.append(Finding("NB005", f"local image does not exist: {target}"))

    return findings


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check selected tutorial notebooks against CONTRIBUTING.md."
    )
    parser.add_argument("notebooks", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    failed = False
    for path in args.notebooks:
        findings = validate_notebook(path)
        if findings:
            failed = True
            print(f"FAIL {path}")
            for finding in findings:
                print(f"  {finding.code} {finding.message}")
        else:
            print(f"PASS {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
