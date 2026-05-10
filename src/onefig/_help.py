from __future__ import annotations

import enum
import shutil
import textwrap
from collections.abc import Sequence
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

try:
    from types import UnionType  # Python 3.10+
except ImportError:  # pragma: no cover
    UnionType = None  # type: ignore[assignment,misc]

_MIN_WIDTH = 60
_MAX_WIDTH = 100
_INDENT = "  "
_META_SEP = "  ·  "


def format_help(model: BaseModel, title: str | None = None) -> str:
    """Render a schema-aware help panel for a :class:`ConfigModel` instance.

    Walks the model recursively and produces a tyro-style boxed panel
    showing each leaf field's dotted path, type, default, current value,
    and docstring/description.

    Args:
        model: A Pydantic model instance to introspect.
        title: Header for the panel. Defaults to the model class name.

    Returns:
        A multi-line string ready to print.
    """
    header = title or type(model).__name__
    width = _resolve_width()
    text_width = width - 4  # exclude '│ ' on the left and ' │' on the right

    intro = textwrap.wrap(
        "Override fields with key=value (or use --show / --help).",
        width=text_width,
    )

    field_section = _build_field_lines(model, text_width)

    flags_section = [
        "Special flags:",
        f"{_INDENT}--show         Print the resolved config and exit.",
        f"{_INDENT}--help, -h     Show this help and exit.",
    ]

    return _render_panel(
        header,
        sections=[intro, field_section, flags_section],
        width=width,
    )


def _resolve_width() -> int:
    term = shutil.get_terminal_size((88, 20)).columns
    return max(_MIN_WIDTH, min(term, _MAX_WIDTH))


def _build_field_lines(model: BaseModel, text_width: int) -> list[str]:
    entries = list(_iter_fields(model))
    if not entries:
        return ["(no fields)"]

    body_width = max(20, text_width - len(_INDENT))
    out: list[str] = []
    for idx, (path, info, current) in enumerate(entries):
        if idx > 0:
            out.append("")
        type_str = _format_type(info.annotation)
        meta_parts: list[str] = []
        default_str = _format_default(info)
        if default_str is not None:
            meta_parts.append(f"default: {default_str}")
        meta_parts.append(f"current: {current!r}")

        out.extend(_wrap_with_indent(f"{path} : {type_str}", body_width, hang="    "))
        out.extend(
            _wrap_with_indent(
                f"({_META_SEP.join(meta_parts)})", body_width, hang="    ", lead="    "
            )
        )
        if info.description:
            for raw in info.description.splitlines():
                out.extend(
                    _wrap_with_indent(raw, body_width, hang="    ", lead="    ")
                    or ["    "]
                )
    return [f"{_INDENT}{line}" if line else "" for line in out]


def _wrap_with_indent(
    text: str, width: int, *, hang: str = "", lead: str = ""
) -> list[str]:
    """Wrap ``text``; first line uses ``lead``, continuation lines use ``hang``."""
    wrapped = textwrap.wrap(
        text,
        width=width,
        initial_indent=lead,
        subsequent_indent=hang,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [lead]


def _render_panel(title: str, sections: Sequence[Sequence[str]], width: int) -> str:
    inner = width - 2  # chars between corner glyphs
    text_width = width - 4  # chars usable for actual content

    title_str = f" {title} "
    dashes = inner - 1 - len(title_str)
    if dashes < 1:
        # Title too long for the chosen width; degrade gracefully.
        out_lines = ["╭" + "─" * inner + "╮", _pad(title.center(text_width), width)]
    else:
        out_lines = ["╭─" + title_str + "─" * dashes + "╮"]

    nonempty_sections = [s for s in sections if s]
    for i, section in enumerate(nonempty_sections):
        if i > 0:
            out_lines.append("├" + "─" * inner + "┤")
        out_lines.append(_pad("", width))
        for raw_line in section:
            out_lines.append(_pad(_truncate(raw_line, text_width), width))
        out_lines.append(_pad("", width))

    out_lines.append("╰" + "─" * inner + "╯")
    return "\n".join(out_lines)


def _pad(content: str, width: int) -> str:
    text_width = width - 4
    return "│ " + content.ljust(text_width) + " │"


def _truncate(line: str, width: int) -> str:
    if len(line) <= width:
        return line
    if width <= 1:
        return line[:width]
    return line[: width - 1] + "…"


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
