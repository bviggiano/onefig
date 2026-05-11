from __future__ import annotations

import sys
from typing import Any

_RED = "\033[31m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"


class _MissingType:
    """Singleton sentinel for keys that are absent from one side of a diff."""

    _instance: _MissingType | None = None

    def __new__(cls) -> _MissingType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _MissingType()


def compute_diff(
    a: dict[str, Any], b: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Diff two flat dotted-key dicts.

    Returns ``{path: (old, new)}`` for every key whose value differs.
    Keys present in only one side use the :data:`MISSING` sentinel for
    the absent side.

    The returned dict is ordered: keys present in ``a`` come first (in
    ``a``'s order), then keys only present in ``b`` (in ``b``'s order).
    Makes diff output stable and easy to scan.
    """
    out: dict[str, tuple[Any, Any]] = {}
    for k, va in a.items():
        vb = b.get(k, MISSING)
        if va != vb:
            out[k] = (va, vb)
    for k, vb in b.items():
        if k in a:
            continue
        out[k] = (MISSING, vb)
    return out


def format_diff(
    diff: dict[str, tuple[Any, Any]],
    *,
    color: bool | None = None,
    empty_message: str = "(no changes)",
) -> str:
    """Render a diff dict as an aligned, human-readable string.

    Layout is ``key  old  →  new`` with the columns padded to a common
    width. ANSI red marks the ``old`` side, green marks the ``new`` side;
    :data:`MISSING` renders dimmed.

    Args:
        diff: Mapping from dotted path to ``(old, new)`` tuple (the
            output of :func:`compute_diff` / :meth:`ConfigModel.diff`).
        color: ``True`` / ``False`` to force ANSI on or off. ``None``
            (default) auto-detects via ``sys.stdout.isatty()``.
        empty_message: Text to return when ``diff`` is empty.

    Returns:
        A multi-line string ready to print.
    """
    if not diff:
        return empty_message

    if color is None:
        color = sys.stdout.isatty()

    raw_olds = [_format_value(old) for old, _ in diff.values()]
    raw_news = [_format_value(new) for _, new in diff.values()]

    key_width = max(len(k) for k in diff)
    old_width = max(len(s) for s in raw_olds)

    lines: list[str] = []
    for (key, (old, new)), old_str, new_str in zip(diff.items(), raw_olds, raw_news):
        old_padded = old_str.ljust(old_width)
        old_styled = _style(old_padded, _RED, old, enabled=color)
        new_styled = _style(new_str, _GREEN, new, enabled=color)
        lines.append(f"  {key.ljust(key_width)}  {old_styled}  →  {new_styled}")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is MISSING:
        return "<MISSING>"
    return repr(value)


def _style(text: str, color_code: str, value: Any, *, enabled: bool) -> str:
    if not enabled:
        return text
    # MISSING dims; everything else uses its side's color.
    code = _DIM if value is MISSING else color_code
    return f"{code}{text}{_RESET}"
