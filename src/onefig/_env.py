from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from onefig._cli import _coerce


def parse_env(
    environ: Mapping[str, str],
    *,
    prefix: str,
    delimiter: str = "__",
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Parse environment variables into a flat override dict.

    Keys that don't start with ``prefix`` are ignored. For matching keys,
    the prefix is stripped, ``delimiter`` is replaced with ``.`` (so a
    POSIX-legal name can address nested fields), and the result is
    lowercased unless ``case_sensitive`` is set. Values are coerced with
    best-effort JSON parsing (same rules as the CLI parser).

    Args:
        environ: Mapping of env var names to string values (typically
            ``os.environ``).
        prefix: Required prefix to scope which env vars are consumed.
            Pass ``""`` to read every variable (rarely what you want).
        delimiter: Substring that separates nested-field segments inside
            an env var name. Defaults to ``"__"``, matching the
            pydantic-settings convention.
        case_sensitive: If ``False`` (default), keys are lowercased after
            stripping the prefix. Set ``True`` for schemas with mixed-case
            field names.

    Returns:
        Flat mapping of override keys (e.g. ``"model.lr"``) to coerced
        values, ready to hand to ``apply_overrides``.

    Raises:
        ValueError: If a matching env var, after stripping the prefix and
            splitting on the delimiter, contains an empty segment.
    """
    out: dict[str, Any] = {}
    for raw_name, raw_value in environ.items():
        if not raw_name.startswith(prefix):
            continue
        tail = raw_name[len(prefix) :]
        if not tail:
            # Var name equals the prefix exactly; nothing to address.
            continue
        segments = tail.split(delimiter)
        if any(seg == "" for seg in segments):
            raise ValueError(
                f"Env var {raw_name!r} produces an empty key segment "
                f"after stripping prefix {prefix!r} and splitting on "
                f"{delimiter!r}."
            )
        key = ".".join(segments)
        if not case_sensitive:
            key = key.lower()
        out[key] = _coerce(raw_value)
    return out
