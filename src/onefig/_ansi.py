"""ANSI escape codes shared by the diff and validation-error renderers."""

from __future__ import annotations

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def style(text: str, code: str, *, enabled: bool) -> str:
    """Wrap ``text`` in ``code`` when ``enabled``; otherwise return it unchanged."""
    return f"{code}{text}{RESET}" if enabled else text
