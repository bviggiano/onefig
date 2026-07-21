from __future__ import annotations

from typing import Any


def format_tree(data: Any, name: str = "Config") -> str:
    """Render a nested ``dict`` / ``list`` as an ASCII tree.

    Args:
        data: The structure to render. Dicts and lists are walked
            recursively; everything else is rendered with :func:`repr`.
        name: Label for the root of the tree.

    Returns:
        A multi-line string with ``├──`` / ``└──`` connectors.
    """
    lines = [f"{name}:"]
    lines.extend(_render(data, prefix=""))
    return "\n".join(lines)


def _render(data: Any, prefix: str) -> list[str]:
    lines: list[str] = []

    if isinstance(data, dict):
        items = list(data.items())
        last = len(items) - 1
        for i, (key, value) in enumerate(items):
            connector = "└── " if i == last else "├── "
            child_prefix = prefix + ("    " if i == last else "│   ")
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{connector}{key}:")
                lines.extend(_render(value, child_prefix))
            else:
                lines.append(f"{prefix}{connector}{key}: {value!r}")

    elif isinstance(data, list):
        last = len(data) - 1
        for i, item in enumerate(data):
            connector = "└── " if i == last else "├── "
            child_prefix = prefix + ("    " if i == last else "│   ")
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{connector}[{i}]:")
                lines.extend(_render(item, child_prefix))
            else:
                lines.append(f"{prefix}{connector}[{i}]: {item!r}")

    return lines


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested ``dict`` into ``{"a.b.c": value}`` form.

    Lists of dicts are indexed (``"a.0.b"``); lists of scalars also get
    index keys (``"a.0"``). Inverse of :func:`unflatten`.

    Args:
        data: Nested mapping to flatten.
        prefix: Internal — leading dotted prefix used during recursion.

    Returns:
        A flat mapping of dotted-path keys to leaf values.
    """
    out: dict[str, Any] = {}
    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(flatten(item, f"{key}.{i}."))
                else:
                    out[f"{key}.{i}"] = item
        else:
            out[key] = v
    return out


def unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`flatten` — reconstruct nested structure from dotted keys.

    Segments composed entirely of digits become list indices; all other
    segments are dict keys. Indices must be dense (``0..n-1``) and may
    appear in any order.

    Limitations:
      * Empty lists/dicts cannot be represented in flat form, so they are
        lost in a round-trip.
      * Dicts that intentionally use integer-string keys (e.g.
        ``{"0": "x"}``) are reconstructed as lists. The flat format cannot
        disambiguate them.

    Args:
        flat: Flat mapping of dotted-path keys to leaf values.

    Returns:
        A nested ``dict`` (with embedded lists where appropriate).

    Raises:
        ValueError: For sparse or non-zero-based list indices, or if the
            flat keys describe a structural conflict (a path is both a
            leaf and a parent).
    """
    root: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            existing = node.get(part)
            if existing is None:
                existing = {}
                node[part] = existing
            elif not isinstance(existing, dict):
                raise ValueError(
                    f"Conflict at {key!r}: {part!r} is both a leaf and a parent."
                )
            node = existing
        leaf = parts[-1]
        if leaf in node:
            raise ValueError(f"Duplicate key {key!r} in flat dict.")
        node[leaf] = value
    return _dicts_to_lists(root)


def _dicts_to_lists(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    converted = {k: _dicts_to_lists(v) for k, v in value.items()}
    if converted and all(k.isdigit() for k in converted):
        indices = sorted(int(k) for k in converted)
        if indices != list(range(len(indices))):
            raise ValueError(f"List indices are sparse or non-zero-based: {indices}.")
        return [converted[str(i)] for i in indices]
    return converted
