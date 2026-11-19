"""Prepare the standalone viewer with its latest Git modification date.

Run ``make build-gravity-viewer`` from the repository root before distributing
the static viewer. The browser itself cannot inspect repository history.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VIEWER_HTML = Path("tools/gravity_viewer/index.html")
VIEWER_HISTORY_PATHS = (
    "tools/gravity_viewer/index.html",
    "tools/gravity_viewer/viewer.js",
)
CREDIT_PATTERN = re.compile(
    r'(<p class="title-credit">By Michel Bierlaire · Updated )([^<]+)(</p>)'
)
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def latest_viewer_commit_date(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Return the latest commit date affecting either viewer source file."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", *VIEWER_HISTORY_PATHS],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit_date = result.stdout.strip()
    if not commit_date:
        raise RuntimeError("Git history has no commit affecting the viewer files.")
    parsed = date.fromisoformat(commit_date)
    return f"{parsed.day} {MONTH_NAMES[parsed.month - 1]} {parsed.year}"


def render_credit(html: str, commit_date: str) -> str:
    """Replace the generated credit date, requiring exactly one credit line."""
    updated, replacements = CREDIT_PATTERN.subn(rf"\g<1>{commit_date}\g<3>", html)
    if replacements != 1:
        raise ValueError(
            "viewer HTML must contain exactly one generated title-credit line; "
            f"found {replacements}."
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if index.html does not already contain the latest viewer commit date",
    )
    arguments = parser.parse_args()

    path = REPOSITORY_ROOT / VIEWER_HTML
    html = path.read_text(encoding="utf-8")
    expected = latest_viewer_commit_date()
    updated = render_credit(html, expected)
    if arguments.check:
        if updated != html:
            raise SystemExit(
                "viewer title credit is stale; run `make build-gravity-viewer`."
            )
        print(f"Viewer credit is current: {expected}.")
        return 0

    if updated != html:
        path.write_text(updated, encoding="utf-8")
    print(f"Prepared gravity viewer with latest modification date: {expected}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
