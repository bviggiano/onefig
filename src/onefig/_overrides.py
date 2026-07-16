from __future__ import annotations

import dataclasses
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
    leaf-name matches. Paths may descend through nested models and through
    plain dataclass fields (e.g. a dataclass held by a discriminated union).

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


def _child_field_names(node: Any) -> list[str] | None:
    """Return the child field names of a structured node, or ``None`` for a leaf.

    Recurses through Pydantic models and plain dataclass instances alike, so
    a dataclass field's subfields are addressable as override paths. Anything
    else (scalars, tuples, lists, mappings) is treated as a leaf.
    """
    if isinstance(node, BaseModel):
        return list(type(node).model_fields)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return [f.name for f in dataclasses.fields(node)]
    return None


def _iter_full_paths(node: Any, prefix: str = ""):
    """Yield the dotted path of every leaf field, reading the live instance.

    Only the values actually present are enumerated, so a discriminated union
    contributes the fields of its current variant.
    """
    names = _child_field_names(node)
    if names is None:
        yield prefix[:-1]  # strip the trailing separator left by the parent
        return
    for name in names:
        yield from _iter_full_paths(getattr(node, name), f"{prefix}{name}.")


def _set_dotted(model: BaseModel, dotted_path: str, value: Any) -> None:
    """Assign ``value`` at ``dotted_path``, routing through the owning model.

    Descends through nested models to the deepest one on the path. If the
    remaining segments address a plain dataclass field, the dataclass value is
    rebuilt with the override merged in and reassigned to its model field, so
    that model's ``validate_assignment`` re-runs coercion and validation on the
    whole field (for a discriminated union, that re-discriminates and re-checks
    the variant). Otherwise the leaf is assigned directly.
    """
    parts = dotted_path.split(".")
    parent: Any = model
    idx = 0
    while idx < len(parts) - 1 and isinstance(getattr(parent, parts[idx]), BaseModel):
        parent = getattr(parent, parts[idx])
        idx += 1

    field_name = parts[idx]
    rest = parts[idx + 1 :]
    if not rest:
        setattr(parent, field_name, value)
        return

    merged = _merge_into(getattr(parent, field_name), rest, value)
    setattr(parent, field_name, merged)


def _merge_into(node: Any, path: list[str], value: Any) -> dict[str, Any]:
    """Return ``node`` as a nested dict with ``path`` set to ``value``.

    ``node`` is a dataclass instance; ``dataclasses.asdict`` gives a plain,
    fully-owned mapping that Pydantic can re-validate on assignment.
    """
    data = dataclasses.asdict(node)
    cursor = data
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return data
