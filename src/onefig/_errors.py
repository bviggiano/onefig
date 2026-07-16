from __future__ import annotations

import dataclasses
import inspect
import os
import re
import sys
import typing
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ValidationError

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_RESET = "\033[0m"

# Prefixes Pydantic prepends when a plain ``ValueError`` / ``AssertionError`` is
# raised inside a validator; they add no information once the field is named.
_MSG_PREFIXES = ("Value error, ", "Assertion failed, ")
_TYPE_NAMES = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}


class ConfigError(ValueError):
    """Raised when configuration data fails validation.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    keep catching it. Its string form is a readable report — one tree of the
    offending fields, each with a plain-language message and, when resolvable,
    the source location of the model or dataclass that defines it — and never
    includes the documentation URLs Pydantic appends.
    """


@contextmanager
def clean_validation_errors(
    root: type | None = None, *, name: str | None = None
) -> Iterator[None]:
    """Re-raise any :class:`pydantic.ValidationError` as a clean :class:`ConfigError`.

    Args:
        root: The model class being validated. When given, each reported field
            is annotated with the source location of the class that defines it.
        name: A label for the report header (e.g. a config file stem). Defaults
            to the root class name, or a generic label.
    """
    try:
        yield
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, root=root, name=name)) from exc


def format_validation_error(
    exc: ValidationError,
    *,
    root: type | None = None,
    name: str | None = None,
    color: bool | None = None,
) -> str:
    """Render a :class:`pydantic.ValidationError` as a readable field tree.

    Each offending field is shown in the same ASCII-tree style as
    :meth:`ConfigModel.display`, with a plain-language message and — when
    ``root`` is supplied and the location resolves — the ``file:line`` of the
    model or dataclass that defines it. The documentation URLs Pydantic appends
    are always removed.

    Args:
        exc: The validation error to render.
        root: The model class that was validated, used to resolve source
            locations. Locations are omitted when ``None``.
        name: Header label. Defaults to the root class name or a generic label.
        color: Force ANSI styling on/off. ``None`` (default) auto-detects via
            ``sys.stderr.isatty()``.

    Returns:
        A multi-line report ready to print.
    """
    try:
        return _format_tree(exc, root=root, name=name, color=color)
    except Exception:
        # Formatting must never mask the underlying error; fall back to the
        # plain report with only the URLs stripped.
        return _strip_urls(exc)


def _strip_urls(exc: ValidationError) -> str:
    return "\n".join(
        line for line in str(exc).splitlines() if "errors.pydantic.dev" not in line
    )


# ---- rendering -------------------------------------------------------------


def _style(text: str, code: str, *, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled else text


def _format_tree(
    exc: ValidationError,
    *,
    root: type | None,
    name: str | None,
    color: bool | None,
) -> str:
    use_color = sys.stderr.isatty() if color is None else color
    errors = exc.errors(include_url=False)
    count = len(errors)
    label = name or getattr(root, "__name__", None)

    problems = f"{count} problem{'s' if count != 1 else ''} found"
    title = "Config Validation Error"
    if label:
        title = f"{title} in {label!r}"
    header = (
        f"{_style('✗', _RED, enabled=use_color)} "
        f"{_style(title, _BOLD, enabled=use_color)} — {problems}"
    )
    tree = _build_tree(errors, root)
    return "\n".join([header, *_render_nodes(tree, "", enabled=use_color)])


def _build_tree(errors: list[Any], root: type | None) -> OrderedDict[str, Any]:
    tree: OrderedDict[str, Any] = OrderedDict()
    for err in errors:
        segments, message, location = _analyze(err, root)
        node = tree
        leaf: dict[str, Any] | None = None
        for key in segments:
            leaf = node.setdefault(key, {"children": OrderedDict(), "error": None})
            node = leaf["children"]
        if leaf is not None:
            leaf["error"] = (message, location)
    return tree


def _render_nodes(
    node: OrderedDict[str, Any], prefix: str, *, enabled: bool
) -> list[str]:
    lines: list[str] = []
    items = list(node.items())
    last = len(items) - 1
    for i, (key, data) in enumerate(items):
        connector = "└── " if i == last else "├── "
        child_prefix = prefix + ("    " if i == last else "│   ")
        error = data["error"]
        if error is not None:
            message, location = error
            mark = _style("✗", _RED, enabled=enabled)
            key_styled = _style(key, _BOLD, enabled=enabled)
            lines.append(f"{prefix}{connector}{key_styled}  {mark} {message}")
            if location is not None:
                pointer = _style(f"→ {location}", _DIM, enabled=enabled)
                lines.append(f"{child_prefix}{pointer}")
        else:
            lines.append(f"{prefix}{connector}{key}")
        lines.extend(_render_nodes(data["children"], child_prefix, enabled=enabled))
    return lines


# ---- message normalization -------------------------------------------------


def _humanize(err: Any) -> str:
    kind = err["type"]
    message = err["msg"]
    given = err.get("input")
    for prefix in _MSG_PREFIXES:
        if message.startswith(prefix):
            message = message[len(prefix) :]
    if kind == "missing":
        return "required field is missing"
    if kind == "extra_forbidden":
        return "unknown field"
    if kind == "union_tag_invalid":
        ctx = err.get("ctx", {})
        tag = ctx.get("tag", "?")
        discriminator = str(ctx.get("discriminator", "tag")).strip("'")
        expected = ctx.get("expected_tags", "")
        return f"{tag!r} is not a valid {discriminator}; expected {expected}"
    if kind.endswith("_parsing") or kind.endswith("_type"):
        want = kind.split("_", 1)[0]
        want = _TYPE_NAMES.get(want, want)
        return f"must be a valid {want} (got {_short(given)})"
    if _is_scalar(given) and repr(given) not in message:
        return f"{message} (got {_short(given)})"
    return message


def _short(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list, tuple))


# ---- source-location resolution --------------------------------------------


def _child_fields(cls: Any) -> dict[str, tuple[Any, Any]] | None:
    """Map a class's field names to ``(annotation, discriminator)`` pairs."""
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return {n: (f.annotation, f.discriminator) for n, f in cls.model_fields.items()}
    if dataclasses.is_dataclass(cls):
        return {f.name: (f.type, None) for f in dataclasses.fields(cls)}
    return None


def _iter_variants(annotation: Any) -> Iterator[type]:
    """Yield the concrete classes of a (possibly Annotated) union annotation."""
    for arg in typing.get_args(annotation):
        if isinstance(arg, type):
            yield arg
        else:
            yield from _iter_variants(arg)


def _variant_for_tag(annotation: Any, tag: str, discriminator: str) -> type | None:
    for variant in _iter_variants(annotation):
        if _discriminator_default(variant, discriminator) == tag:
            return variant
    return None


def _discriminator_default(variant: type, name: str) -> Any:
    if issubclass(variant, BaseModel):
        info = variant.model_fields.get(name)
        return info.default if info is not None else None
    if dataclasses.is_dataclass(variant):
        for field in dataclasses.fields(variant):
            if field.name == name:
                return field.default
    return None


def _analyze(err: Any, root: type | None) -> tuple[list[str], str, str | None]:
    """Resolve one error into (display segments, message, source location).

    A discriminated-union tag is folded into the preceding field's label
    (``optimizer (adamw)``) rather than shown as its own level, and when the
    message names a field of the offending model, that field is appended as the
    final segment so the branch reaches the exact leaf.
    """
    message = _humanize(err)
    if root is None:
        return [str(seg) for seg in err["loc"]], message, None

    display, owner, field_name = _walk(root, err["loc"])
    location = _source(owner, field_name)

    container = owner if field_name is None else _field_type(owner, field_name)
    fields = _child_fields(container) if container is not None else None
    if container is not None and fields and message:
        leaf = message.split(None, 1)[0]
        if leaf in fields:
            display = [*display, leaf]
            message = message[len(leaf) :].lstrip() or message
            drilled = _source(container, leaf)
            if drilled is not None:
                location = drilled
    return display, message, location


def _walk(root: type, loc: tuple[Any, ...]) -> tuple[list[str], type, str | None]:
    """Walk ``loc`` from ``root``, returning display labels, owner, leaf field.

    Nested models and dataclasses are descended; a discriminated-union tag is
    folded into the preceding label. Unknown segments (e.g. an extra field) stop
    the walk, leaving the enclosing class as the owner.
    """
    display: list[str] = []
    current: Any = root
    owner: type = root
    field_name: str | None = None
    pending_union: tuple[Any, str] | None = None
    for seg in loc:
        segment = str(seg)
        if pending_union is not None:
            if display:
                display[-1] = f"{display[-1]} ({segment})"
            variant = _variant_for_tag(pending_union[0], segment, pending_union[1])
            if variant is not None:
                current = owner = variant
                field_name = None
            pending_union = None
            continue
        display.append(segment)
        fields = _child_fields(current)
        if fields and segment in fields:
            owner = current
            field_name = segment
            annotation, discriminator = fields[segment]
            if discriminator:
                pending_union = (annotation, discriminator)
                current = None
            elif isinstance(annotation, type):
                current = annotation
        else:
            field_name = segment
            break
    return display, owner, field_name


def _field_type(owner: type, field_name: str) -> type | None:
    """Return the class a field is typed as, if it is a model or dataclass."""
    fields = _child_fields(owner)
    if not fields or field_name not in fields:
        return None
    annotation = fields[field_name][0]
    if isinstance(annotation, type) and (
        issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation)
    ):
        return annotation
    return None


def _source(owner: type, field_name: str | None) -> str | None:
    try:
        source = inspect.getsourcefile(owner)
        lines, start = inspect.getsourcelines(owner)
    except (TypeError, OSError):
        return None
    if source is None:
        return None
    line = start
    if field_name:
        pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*[:=]")
        for offset, text in enumerate(lines):
            if pattern.match(text):
                line = start + offset
                break
    return f"{_relative(source)}:{line}"


def _relative(path: str) -> str:
    try:
        relative = os.path.relpath(path)
    except ValueError:
        return path
    return relative if not relative.startswith("..") else path
