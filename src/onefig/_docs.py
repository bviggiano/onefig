"""Field documentation: each field's docstring, and a table of a class's fields.

:func:`field_docs` reads what each field of a config or dataclass says about
itself: a config field's description (its docstring, or ``description=``), and the
string literal just below a dataclass field. :func:`describe_fields` lays the
fields out as a table, which renders as Markdown in a notebook and prints as
Markdown text anywhere else.

Table cells are escaped the way Markdown tables need: a pipe in prose is
backslash-escaped, and code (field names, types, defaults, and double-backtick
spans in docstrings) is written as HTML ``<code>`` with the pipe, ``&``, ``<`` and
``>`` encoded, because Markdown code spans ignore backslash escapes and a type such
as ``int | None`` would otherwise split its cell.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import textwrap
from typing import Any

from pydantic import BaseModel

from onefig._help import _format_type, _format_value
from onefig._traits import trait_text, traits_of

_RST_CODE = re.compile(r"``(.+?)``")


def field_docs(cls: type) -> dict[str, str]:
    """Return each field's docstring of a config or dataclass, in field order.

    For a pydantic model (a :class:`ConfigModel` included) it is the field's
    description: its docstring, or ``Field(description=...)``. For a dataclass it
    is the string literal just below the field; a subclass's docstring for an
    inherited field replaces the base's. Fields without one are left out.

    Raises:
        TypeError: If ``cls`` is neither a pydantic model nor a dataclass.
    """
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return {
            name: info.description
            for name, info in cls.model_fields.items()
            if info.description
        }
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(
            f"field_docs takes a pydantic model or a dataclass; got {cls!r}."
        )
    docs: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if dataclasses.is_dataclass(klass):
            docs.update(_attribute_docs(klass))
    return {f.name: docs[f.name] for f in dataclasses.fields(cls) if f.name in docs}


class FieldTable:
    """A table of a class's fields, shown as Markdown in a notebook.

    ``str(table)`` is the Markdown text; ``rows`` holds the cells, unescaped.
    """

    def __init__(self, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
        self.header = header
        self.rows = rows

    def __str__(self) -> str:
        lines = [
            "| " + " | ".join(self.header) + " |",
            "|" + "|".join("---" for _ in self.header) + "|",
        ]
        for row in self.rows:
            name, kind, *middle, description = row
            cells = [code(name), code(kind), *(code(m) for m in middle)]
            lines.append("| " + " | ".join([*cells, prose(description)]) + " |")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return str(self)


def describe_fields(obj: Any) -> FieldTable:
    """Return a table of the fields of a config or dataclass, or an instance of one.

    Each row is a field: its name, its type and its description, from
    :func:`field_docs`. A config's table also has each field's default, and a field
    trait's :meth:`~onefig.FieldTrait.format_type`,
    :meth:`~onefig.FieldTrait.format_value` and
    :meth:`~onefig.FieldTrait.describe` text are used as in ``--help``. For a
    dataclass instance, a field holding an array is typed by its dtype (``str``,
    ``int32``, ``float64``).

    Raises:
        TypeError: If ``obj`` is neither a pydantic model nor a dataclass, nor an
            instance of one.
    """
    cls: Any = obj if isinstance(obj, type) else type(obj)
    docs = field_docs(cls)  # raises unless cls is a pydantic model or a dataclass
    if issubclass(cls, BaseModel):
        return _config_table(cls, docs)
    instance = None if isinstance(obj, type) else obj
    rows: list[tuple[str, ...]] = []
    for field in dataclasses.fields(cls):
        value = getattr(instance, field.name) if instance is not None else None
        kind = _dtype_name(value) or _annotation_name(field.type)
        rows.append((field.name, kind, docs.get(field.name, "")))
    return FieldTable(("Field", "Type", "Description"), rows)


def code(text: str) -> str:
    """``text`` as inline code that survives a Markdown table cell."""
    for char, entity in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ("|", "&#124;")):
        text = text.replace(char, entity)
    return f"<code>{text}</code>"


def prose(text: str) -> str:
    """``text`` as one Markdown table cell: lines joined, pipes escaped, and
    double-backtick spans written as :func:`code`."""
    joined = " ".join(text.split())
    parts = _RST_CODE.split(joined)  # odd entries are the code spans
    return "".join(
        code(part) if i % 2 else part.replace("|", "\\|")
        for i, part in enumerate(parts)
    )


def _config_table(cls: type[BaseModel], docs: dict[str, str]) -> FieldTable:
    rows: list[tuple[str, ...]] = []
    for name, info in cls.model_fields.items():
        traits = traits_of(info)
        kind = trait_text(traits, "format_type", info.annotation)
        kind = kind or _format_type(info.annotation)
        if info.is_required():
            default = "required"
        else:
            value = info.get_default(call_default_factory=True)
            default = trait_text(traits, "format_value", value) or _format_value(value)
        notes = [docs.get(name), *(t.describe() for t in traits)]
        rows.append((name, kind, default, " ".join(n for n in notes if n)))
    return FieldTable(("Field", "Type", "Default", "Description"), rows)


def _dtype_name(value: Any) -> str | None:
    """A readable name for an array's dtype, or ``None`` if ``value`` has none."""
    dtype: Any = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if not isinstance(kind, str):
        return None
    if kind in "USO":
        return "str"
    if kind == "b":
        return "bool"
    return str(getattr(dtype, "name", dtype))


def _annotation_name(annotation: Any) -> str:
    if isinstance(annotation, str):  # unevaluated, under postponed annotations
        return annotation
    return _format_type(annotation)


def _attribute_docs(cls: type) -> dict[str, str]:
    """The docstrings below the annotated attributes in ``cls``'s own body."""
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):  # no source, as for a class defined interactively
        return {}
    body = ast.parse(source).body[0]
    if not isinstance(body, ast.ClassDef):
        return {}
    docs: dict[str, str] = {}
    for node, after in zip(body.body, body.body[1:]):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(after, ast.Expr)
            and isinstance(after.value, ast.Constant)
            and isinstance(after.value.value, str)
        ):
            docs[node.target.id] = inspect.cleandoc(after.value.value)
    return docs
