from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel

from onefig._errors import FROZEN_HINT, FrozenConfigError
from onefig._union import UnionInfo, union_info


def apply_overrides(
    model: BaseModel,
    args: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Update ``model`` in place with values from a flat override mapping.

    Keys may be full dotted paths (e.g. ``"optimizer.lr"``) or unambiguous
    suffix subpaths, including the bare leaf (e.g. ``"lr"``). Shorter/base
    paths take precedence. Paths may descend through nested models and through
    plain dataclass fields (e.g. a dataclass held by a tagged union).

    A tagged union's tag path (e.g. ``"optimizer.name"``) switches the variant:
    a new tag resets that node to the new variant's defaults, and the other keys
    resolve against the new variant, so ``optimizer.name=sgd momentum=0.5``
    works in either order. Re-sending the current tag keeps the node's values.

    All overrides are merged into the config and validated in a **single
    pass**, so a ``model_validator`` (e.g. a cross-field ``lo < hi`` check)
    always sees the fully-applied config, never a half-updated intermediate.
    The outcome is therefore independent of the order the keys are given.

    Args:
        model: A Pydantic model instance to mutate.
        args: Flat mapping of override keys to values.
        strict: If ``True`` (default), unknown keys raise. If ``False``,
            unknown keys are silently skipped.

    Raises:
        ValueError: If a leaf-name key matches multiple paths, or (when
            ``strict``) a key matches no path.
        FrozenConfigError: If ``model`` is frozen.
        pydantic.ValidationError: If the merged config fails validation,
            including a switch to an unknown tag.
    """
    unions = _tag_paths(model)
    switches: dict[str, Any] = {}
    rest: dict[str, Any] = {}
    candidates = resolve_keys(model)
    for key, value in args.items():
        paths = candidates.get(key)
        if paths is not None and len(paths) == 1 and paths[0] in unions:
            switches[paths[0]] = value
        else:
            rest[key] = value

    if switches:
        _check_unfrozen(model)
    data = model.model_dump()
    switched: dict[str, type] = {}
    for path in sorted(switches, key=lambda p: p.count(".")):
        node_path, tag = path.rsplit(".", 1)
        new_tag = switches[path]
        if _current_tag(data, node_path, tag) != new_tag:
            _assign_in_mapping(data, node_path.split("."), {tag: new_tag})
            variant = unions[path].variants.get(new_tag)
            if variant is None:  # unknown tag: report it as the validation error
                type(model).model_validate(data)
            else:
                switched[node_path] = variant

    resolved = _resolve_override_paths(
        resolve_keys(model, switched=switched) if switched else candidates,
        rest,
        strict=strict,
    )
    if not resolved and not switches:
        return
    _check_unfrozen(model)
    # Merge every override into a plain mapping, then re-validate the whole config
    # once, so cross-field validators run against the final merged state and the
    # result does not depend on override order.
    for path, value in resolved:
        _assign_in_mapping(data, path.split("."), value)
    validated = type(model).model_validate(data)
    for name in type(model).model_fields:
        object.__setattr__(model, name, getattr(validated, name))
    fields_set = set(validated.__pydantic_fields_set__)
    object.__setattr__(model, "__pydantic_fields_set__", fields_set)


def _check_unfrozen(model: BaseModel) -> None:
    if getattr(model, "_frozen", False):
        raise FrozenConfigError(
            f"Cannot apply overrides: config is frozen. {FROZEN_HINT}"
        )


def _resolve_override_paths(
    candidates: dict[str, list[str]], args: dict[str, Any], *, strict: bool
) -> list[tuple[str, Any]]:
    """Resolve each override key to its single full dotted path.

    Returns ``(path, value)`` pairs. Raises on an ambiguous leaf key, or (when
    ``strict``) an unknown key; otherwise skips unknown keys.
    """
    resolved: list[tuple[str, Any]] = []
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
        resolved.append((paths[0], value))
    return resolved


def _get_in_mapping(data: dict[str, Any], path: list[str]) -> Any:
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _current_tag(data: dict[str, Any], node_path: str, tag: str) -> Any:
    node = _get_in_mapping(data, node_path.split("."))
    return node.get(tag) if isinstance(node, dict) else None


def _assign_in_mapping(data: dict[str, Any], path: list[str], value: Any) -> None:
    """Set ``value`` at the nested ``path`` inside the mapping ``data``, in place.

    Missing or unset (``None``) intermediate nodes become empty mappings, so a
    path into a just-switched variant, or an unset optional model, can be set.
    """
    cursor = data
    for key in path[:-1]:
        if not isinstance(cursor.get(key), dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def resolve_keys(
    model: BaseModel, *, switched: Mapping[str, type] | None = None
) -> dict[str, list[str]]:
    """Map every accepted user-input key to its candidate full paths.

    Each full dotted path maps to itself. Each *suffix subpath* of a full path
    (its trailing segments, e.g. ``c`` and ``b.c`` for ``a.b.c``) also maps to
    it, unless the suffix is itself a full path -- then that shorter/base path
    takes precedence. A suffix shared by more than one full path is ambiguous
    (``len > 1``) and fails at apply time.

    Args:
        model: Pydantic model whose fields define the path namespace.
        switched: Node paths whose tagged-union variant is being switched,
            mapped to the new variant class; their paths come from that class
            instead of the live value.

    Returns:
        Mapping from override key to a list of candidate dotted paths
        (``len == 1`` means unambiguous).
    """
    full_paths = [path for path, _ in _iter_full_paths(model, "", switched or {})]
    full_set = set(full_paths)

    out: dict[str, list[str]] = {p: [p] for p in full_paths}
    for path in full_paths:
        segments = path.split(".")
        for start in range(1, len(segments)):  # every proper trailing suffix
            suffix = ".".join(segments[start:])
            if suffix in full_set:
                continue  # a shorter/base path owns this key; it wins
            out.setdefault(suffix, []).append(path)
    return out


def _tag_paths(model: BaseModel) -> dict[str, UnionInfo]:
    """Map the tag path of every tagged-union field (``optimizer.name``) to its info."""
    return {path: info for path, info in _iter_full_paths(model, "", {}) if info}


_LEAF = object()


def _children(node: Any) -> list[tuple[str, Any, UnionInfo | None]] | None:
    """Return ``(name, child, union info)`` per field of a structured node.

    A node is a live model or dataclass instance, or a model or dataclass *class*
    (a just-switched variant, whose children come from its field annotations).
    Anything else is a leaf (``None``).
    """
    if isinstance(node, type):
        if issubclass(node, BaseModel):
            return [
                (name, _class_child(f.annotation), union_info(f))
                for name, f in node.model_fields.items()
            ]
        if dataclasses.is_dataclass(node):
            return [
                (f.name, _class_child(f.type), union_info(f.type))
                for f in dataclasses.fields(node)
            ]
        return None
    if isinstance(node, BaseModel):
        return [
            (name, getattr(node, name), union_info(f))
            for name, f in type(node).model_fields.items()
        ]
    if dataclasses.is_dataclass(node):
        return [
            (f.name, getattr(node, f.name), union_info(f.type))
            for f in dataclasses.fields(node)
        ]
    return None


def _class_child(annotation: Any) -> Any:
    """A nested model or dataclass class is walked; any other annotation is a leaf."""
    if isinstance(annotation, type) and (
        issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation)
    ):
        return annotation
    return _LEAF


def _iter_full_paths(
    node: Any, prefix: str, switched: Mapping[str, type]
) -> Iterator[tuple[str, UnionInfo | None]]:
    """Yield ``(path, info)`` for every leaf and every tagged-union tag path.

    Values are read from the live instance, so a tagged union contributes the
    fields of its current variant (or of the variant in ``switched``), plus its
    tag path, whose ``info`` is the union's :class:`UnionInfo`.
    """
    children = _children(node)
    if children is None:
        yield prefix[:-1], None  # strip the trailing separator left by the parent
        return
    for name, child, info in children:
        path = f"{prefix}{name}"
        if info is not None:
            yield f"{path}.{info.tag}", info
            child = switched.get(path, child)
            if child is None or child is _LEAF:
                continue  # an unset optional union, or one inside a switched class
        yield from _iter_full_paths(child, f"{path}.", switched)
