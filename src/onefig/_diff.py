from __future__ import annotations

import sys
from typing import Any

_RED = "\033[31m"
_GREEN = "\033[32m"
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

    Layout follows unified-diff conventions:

    * Changed rows show ``  key  old  →  new`` with red old / green new.
    * Added rows (key only on the new side) show ``+ key  new`` in green
      and have no arrow.
    * Removed rows (key only on the old side) show ``- key  old`` in red
      and have no arrow.

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

    key_width = max(len(k) for k in diff)
    # Compute the old-column width only over rows that have both sides;
    # added/removed rows don't share that column.
    changed_olds = [
        repr(old)
        for old, new in diff.values()
        if old is not MISSING and new is not MISSING
    ]
    old_width = max((len(s) for s in changed_olds), default=0)

    lines: list[str] = []
    for key, (old, new) in diff.items():
        key_col = key.ljust(key_width)
        if old is MISSING:
            new_styled = _style(repr(new), _GREEN, enabled=color)
            lines.append(f"+ {key_col}  {new_styled}")
        elif new is MISSING:
            old_styled = _style(repr(old), _RED, enabled=color)
            lines.append(f"- {key_col}  {old_styled}")
        else:
            # Changed rows: leave the prior value uncolored (it's a
            # reference point, not a removal). Only the new value is
            # highlighted in green. Red is reserved for the `-` rows
            # above so the eye distinguishes "changed" from "removed".
            old_padded = repr(old).ljust(old_width)
            new_styled = _style(repr(new), _GREEN, enabled=color)
            lines.append(f"  {key_col}  {old_padded}  →  {new_styled}")
    return "\n".join(lines)


def _style(text: str, color_code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color_code}{text}{_RESET}"
