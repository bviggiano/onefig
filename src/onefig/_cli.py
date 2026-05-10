from __future__ import annotations

import json
from typing import Any


def parse_overrides(tokens: list[str]) -> dict[str, Any]:
    """Parse a list of ``key=value`` tokens into a flat override dict.

    Values are coerced with best-effort JSON parsing (so ``5`` → int, ``5.0``
    → float, ``true`` / ``false`` → bool, ``null`` → ``None``, ``[1,2]`` →
    list) and fall back to the raw string. Pydantic re-validates at
    assignment, so the coercion is a hint, not a contract.

    Args:
        tokens: List of CLI tokens, each of the form ``key=value``.

    Returns:
        Flat mapping of keys to coerced values.

    Raises:
        ValueError: If a token has no ``=``, or has an empty key.
    """
    out: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Bad override token {tok!r}: expected key=value.")
        key, _, raw = tok.partition("=")
        if not key:
            raise ValueError(f"Bad override token {tok!r}: empty key.")
        out[key] = _coerce(raw)
    return out


def _coerce(raw: str) -> Any:
    if raw == "None":
        return None
    try:
        return json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return raw
