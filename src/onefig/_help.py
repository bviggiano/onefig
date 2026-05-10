from __future__ import annotations

import enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

try:
    from types import UnionType  # Python 3.10+
except ImportError:  # pragma: no cover
    UnionType = None  # type: ignore[assignment,misc]


def format_help(model: BaseModel, title: str | None = None) -> str:
    """Render a schema-aware help string for a :class:`ConfigModel` instance.

    Walks the model recursively and produces one block per leaf field
    (non-:class:`ConfigModel` value), showing dotted path, type, default,
    current value, and the field's docstring/description.

    Args:
        model: A Pydantic model instance to introspect.
        title: Header for the help block. Defaults to the model class name.

    Returns:
        A multi-line string ready to print.
    """
    header = title or type(model).__name__
    lines: list[str] = [header, ""]
    lines.append("Override fields with key=value (or use --show / --help).")
    lines.append("")
    lines.append("Fields:")

    entries = list(_iter_fields(model))
    if not entries:
        lines.append("  (no fields)")
    for path, info, current in entries:
        type_str = _format_type(info.annotation)
        default_str = _format_default(info)
        current_str = repr(current)
        desc = info.description

        head = f"  {path} : {type_str}"
        meta_parts: list[str] = []
        if default_str is not None:
            meta_parts.append(f"default: {default_str}")
        meta_parts.append(f"current: {current_str}")
        head += f"  ({', '.join(meta_parts)})"
        lines.append(head)
        if desc:
            for desc_line in desc.splitlines():
                lines.append(f"      {desc_line}")
        lines.append("")

    lines.append("Special flags:")
    lines.append("  --show         Print the resolved config and exit.")
    lines.append("  --help, -h     Show this help and exit.")
    return "\n".join(lines)


def _iter_fields(
    model: BaseModel, prefix: str = ""
) -> list[tuple[str, FieldInfo, Any]]:
    out: list[tuple[str, FieldInfo, Any]] = []
    for name, info in type(model).model_fields.items():
        full = f"{prefix}{name}"
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            out.extend(_iter_fields(value, full + "."))
        else:
            out.append((full, info, value))
    return out


def _format_default(info: FieldInfo) -> str | None:
    if info.is_required():
        return None
    factory = info.default_factory
    if factory is not None:
        try:
            value = factory()  # type: ignore[call-arg]
        except TypeError:
            return "<factory>"
    else:
        value = info.default
    return repr(value)


def _format_type(annotation: Any) -> str:
    if annotation is None or annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return "{" + ", ".join(repr(a) for a in args) + "}"

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        members = ", ".join(repr(m.value) for m in annotation)
        return f"{annotation.__name__}{{{members}}}"

    if origin is Union or (UnionType is not None and origin is UnionType):
        return " | ".join(_format_type(a) for a in args)

    if origin is not None:
        origin_name = _origin_name(origin)
        if args:
            return f"{origin_name}[{', '.join(_format_type(a) for a in args)}]"
        return origin_name

    if isinstance(annotation, type):
        return annotation.__name__

    return repr(annotation)


def _origin_name(origin: Any) -> str:
    if hasattr(origin, "__name__"):
        return origin.__name__
    return str(origin).replace("typing.", "")
