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
    """Render a diff dict as an aligned ``old → new`` table.

    Every row uses the side-by-side ``key  old  →  new`` layout, with
    ANSI red on the old side, green on the new side, and dimmed
    ``<MISSING>`` for any side without a value (cross-schema diffs
    against partial dicts).

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
    for (key, (old, new)), old_str, new_str in zip(
        diff.items(), raw_olds, raw_news
    ):
        old_padded = old_str.ljust(old_width)
        old_styled = _style_value(old_padded, old, _RED, enabled=color)
        new_styled = _style_value(new_str, new, _GREEN, enabled=color)
        lines.append(f"  {key.ljust(key_width)}  {old_styled}  →  {new_styled}")
    return "\n".join(lines)


def format_against_defaults(
    current: dict[str, Any],
    defaults: dict[str, Any],
    *,
    color: bool | None = None,
    empty_message: str = "(empty config)",
) -> str:
    """Render a config's full state with override highlights.

    Walks every key in ``current``. For fields that match ``defaults``,
    the value renders alone in green (it's the active value, untouched).
    For fields that diverge, the row renders as ``default → current``
    with the default in red and the current in green.

    Useful as a "what does this run actually look like, and where did I
    deviate from defaults?" snapshot.

    Args:
        current: Flat dotted-key dict of the config's current values.
        defaults: Flat dotted-key dict of the schema's default values.
        color: ``True`` / ``False`` to force ANSI on or off. ``None``
            (default) auto-detects via ``sys.stdout.isatty()``.
        empty_message: Text to return when ``current`` is empty.

    Returns:
        A multi-line string ready to print.
    """
    if not current:
        return empty_message

    if color is None:
        color = sys.stdout.isatty()

    keys = list(current.keys())
    key_width = max(len(k) for k in keys)

    overridden_olds = [
        repr(defaults[k])
        for k in keys
        if k in defaults and defaults[k] != current[k]
    ]
    old_width = max((len(s) for s in overridden_olds), default=0)

    lines: list[str] = []
    for key in keys:
        key_col = key.ljust(key_width)
        cur_val = current[key]
        cur_repr = repr(cur_val)
        if key in defaults and defaults[key] == cur_val:
            # Unchanged — single green value, no arrow.
            value_styled = _style(cur_repr, _GREEN, enabled=color)
            lines.append(f"  {key_col}  {value_styled}")
        else:
            # Overridden — default in red, current in green.
            default_repr = (
                repr(defaults[key]) if key in defaults else "<MISSING>"
            )
            old_padded = default_repr.ljust(old_width)
            old_styled = _style(old_padded, _RED, enabled=color)
            new_styled = _style(cur_repr, _GREEN, enabled=color)
            lines.append(f"  {key_col}  {old_styled}  →  {new_styled}")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is MISSING:
        return "<MISSING>"
    return repr(value)


def _style_value(
    text: str, value: Any, color_code: str, *, enabled: bool
) -> str:
    """Style a value's rendered text. MISSING dims; everything else uses
    its side's color."""
    if not enabled:
        return text
    code = _DIM if value is MISSING else color_code
    return f"{code}{text}{_RESET}"


def _style(text: str, color_code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color_code}{text}{_RESET}"
