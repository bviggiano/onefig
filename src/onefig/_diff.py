from __future__ import annotations

from typing import Any


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
