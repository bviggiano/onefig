from __future__ import annotations

import enum
import shutil
import textwrap
from collections.abc import Sequence
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from onefig._overrides import _resolve_keys

try:
    from types import UnionType  # Python 3.10+
except ImportError:  # pragma: no cover
    UnionType = None  # type: ignore[assignment,misc]

_MIN_WIDTH = 60
_MAX_WIDTH = 100
_INDENT = "  "
_HANG = "    "
_META_SEP = "  ·  "

# (local_name, full_path, info, current_value)
FieldEntry = tuple[str, str, FieldInfo, Any]


def format_help(model: BaseModel, title: str | None = None) -> str:
    """Render a schema-aware help block for a :class:`ConfigModel` instance.

    Each nested :class:`ConfigModel` becomes its own boxed sub-panel titled
    by its dotted path; the top panel lists scalar fields directly on the
    model. When a leaf field name is unambiguous as a CLI override, it is
    shown as ``<leaf> (<full.path>)``; otherwise the full dotted path is
    shown alone.

    Args:
        model: A Pydantic model instance to introspect.
        title: Header for the top panel. Defaults to the model class name.

    Returns:
        A multi-line string ready to print.
    """
    header = title or type(model).__name__
    width = _resolve_width()
    text_width = width - 4  # exclude '│ ' on the left and ' │' on the right

    groups = _collect_groups(model)
    leaf_unique = _leaf_uniqueness(model)

    panels: list[str] = []

    intro = textwrap.wrap(
        "Override fields with key=value (or use --show / --help).",
        width=text_width,
    )

    top_path, top_fields = groups[0]
    top_lines = _render_field_lines(top_fields, leaf_unique, text_width)
    top_sections: list[Sequence[str]] = [intro]
    if top_lines:
        top_sections.append(top_lines)
    panels.append(_render_panel(header, sections=top_sections, width=width))

    for path, fields in groups[1:]:
        if not fields:
            continue
        section = _render_field_lines(fields, leaf_unique, text_width)
        panels.append(_render_panel(path, sections=[section], width=width))

    flags = [
        "--show         Print the resolved config and exit.",
        "--help, -h     Show this help and exit.",
    ]
    panels.append(_render_panel("flags", sections=[flags], width=width))

    return "\n\n".join(panels)


def _resolve_width() -> int:
    term = shutil.get_terminal_size((88, 20)).columns
    return max(_MIN_WIDTH, min(term, _MAX_WIDTH))


def _collect_groups(model: BaseModel) -> list[tuple[str, list[FieldEntry]]]:
    """Walk the model and return (panel_title, [field entries]) in declaration order.

    The first group has an empty title and contains the top-level scalar
    fields; subsequent groups are nested ``ConfigModel`` sub-trees keyed by
    their dotted path.
    """
    groups: list[tuple[str, list[FieldEntry]]] = []
    _walk(model, prefix="", groups=groups)
    return groups


def _walk(
    model: BaseModel, prefix: str, groups: list[tuple[str, list[FieldEntry]]]
) -> None:
    local: list[FieldEntry] = []
    nested: list[tuple[str, BaseModel]] = []
    for name, info in type(model).model_fields.items():
        full = f"{prefix}{name}"
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            nested.append((full, value))
        else:
            local.append((name, full, info, value))
    groups.append((prefix.rstrip("."), local))
    for full, child in nested:
        _walk(child, full + ".", groups)


def _leaf_uniqueness(model: BaseModel) -> dict[str, bool]:
    """Map every leaf name to whether it resolves uniquely as a CLI shortcut."""
    candidates = _resolve_keys(model)
    return {key: len(paths) == 1 for key, paths in candidates.items()}


def _render_field_lines(
    fields: Sequence[FieldEntry],
    leaf_unique: dict[str, bool],
    text_width: int,
) -> list[str]:
    body_width = max(20, text_width - len(_INDENT))
    out: list[str] = []
    for idx, (name, full, info, current) in enumerate(fields):
        if idx > 0:
            out.append("")
        type_str = _format_type(info.annotation)
        meta_parts: list[str] = []
        default_str = _format_default(info)
        if default_str is not None:
            meta_parts.append(f"default: {default_str}")
        meta_parts.append(f"current: {current!r}")
        meta_str = f"({_META_SEP.join(meta_parts)})"

        if name == full or not leaf_unique.get(name, False):
            label = full
        else:
            label = f"{name} ({full})"

        out.extend(
            _render_entry(label, type_str, meta_str, info.description, body_width)
        )
    return [f"{_INDENT}{line}" if line else "" for line in out]


def _render_entry(
    label: str,
    type_str: str,
    meta_str: str,
    description: str | None,
    body_width: int,
) -> list[str]:
    """Lay out one field entry.

    Head sits on the first line (with metadata inline when it fits, on its
    own hang-indented line otherwise). The description always starts on a
    new line, hang-indented, and overflow continues at the same indent.
    """
    cont_width = max(20, body_width - len(_HANG))

    head_full = f"{label} : {type_str}  {meta_str}"
    head_short = f"{label} : {type_str}"

    if len(head_full) <= body_width:
        out = [head_full]
    elif len(head_short) <= body_width:
        out = [head_short, f"{_HANG}{meta_str}"]
    else:
        out = textwrap.wrap(
            head_full,
            width=body_width,
            subsequent_indent=_HANG,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [head_full]

    if description:
        desc = " ".join(description.splitlines()).strip()
        if desc:
            desc_lines = textwrap.wrap(
                desc,
                width=cont_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            out.extend(f"{_HANG}{cl}" for cl in desc_lines)

    return out


def _render_panel(title: str, sections: Sequence[Sequence[str]], width: int) -> str:
    inner = width - 2  # chars between corner glyphs
    text_width = width - 4  # chars usable for actual content

    title_str = f" {title} "
    dashes = inner - 1 - len(title_str)
    if dashes < 1:
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
