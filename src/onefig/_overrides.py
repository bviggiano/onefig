from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def apply_overrides(
    model: BaseModel,
    args: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Update ``model`` in place with values from a flat override mapping.

    Keys may be full dotted paths (e.g. ``"optimizer.lr"``) or unambiguous
    leaf names (e.g. ``"lr"``). Full-path matches take precedence over
    leaf-name matches.

    Args:
        model: A Pydantic model instance to mutate.
        args: Flat mapping of override keys to values.
        strict: If ``True`` (default), unknown keys raise. If ``False``,
            unknown keys are silently skipped.

    Raises:
        ValueError: If a leaf-name key matches multiple paths, or (when
            ``strict``) a key matches no path.
        pydantic.ValidationError: If a value fails type validation at the
            destination field.
    """
    candidates = _resolve_keys(model)
    for key, value in args.items():
        paths = candidates.get(key)
        if paths is None:
            if strict:
                raise ValueError(f"Override key {key!r} not found in config schema.")
            continue
        if len(paths) > 1:
            conflicts = ", ".join(paths)
            raise ValueError(
                f"Ambiguous override key {key!r}: matches {conflicts}. "
                "Use the full dotted path instead."
            )
        _set_dotted(model, paths[0], value)


def _resolve_keys(model: BaseModel) -> dict[str, list[str]]:
    """Map every accepted user-input key to its candidate full paths.

    Each full dotted path maps to itself (one candidate). Each leaf name
    maps to all paths ending in it, *unless* the leaf is itself a full
    path — in which case the literal full-path resolution wins.

    Args:
        model: Pydantic model whose fields define the path namespace.

    Returns:
        Mapping from override key to a list of candidate dotted paths
        (``len == 1`` means unambiguous).
    """
    full_paths = list(_iter_full_paths(model))
    full_set = set(full_paths)

    out: dict[str, list[str]] = {p: [p] for p in full_paths}
    for path in full_paths:
        leaf = path.rsplit(".", 1)[-1]
        if leaf == path or leaf in full_set:
            continue  # leaf is its own full path; full-path match takes precedence
        out.setdefault(leaf, []).append(path)
    return out


def _iter_full_paths(model: BaseModel, prefix: str = ""):
    for name in type(model).model_fields:
        full = f"{prefix}{name}"
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            yield from _iter_full_paths(value, full + ".")
        else:
            yield full


def _set_dotted(model: BaseModel, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    obj: Any = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
