"""Field markers: typed objects a project attaches to config fields.

onefig does not interpret a marker. It carries the marker on the field, finds every
field holding a given kind of marker with :meth:`ConfigModel.find`, and asks the
marker how the field should look wherever a config is printed. What a marker means,
such as a range to draw the field's value from, is up to the project that defines
it, and so is how its fields are shown: that lives on the marker class, never on the
configs that use it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic.fields import FieldInfo

from onefig._diff import MISSING


class FieldMarker:
    """Base class for markers attached to config fields with :func:`Field`.

    Subclasses are usually frozen dataclasses. A marker can also be attached
    through ``Annotated[int, MyMarker(...)]``.

    Three hooks set how a marked field is printed. Each returns ``None`` to leave
    onefig's own rendering, so a marker overrides only what it needs; when a field
    has several markers, the first that returns text wins (``describe`` texts are
    all shown).
    """

    def describe(self) -> str | None:
        """Text shown after the field's description in ``--help``."""
        return None

    def format_value(self, value: Any) -> str | None:
        """How a value of the field is shown: the current and default values in
        ``--help``, the field's leaf in :meth:`ConfigModel.display`, and both sides
        of the diff printers."""
        return None

    def format_type(self, annotation: Any) -> str | None:
        """How the field's type is shown in ``--help``."""
        return None


M = TypeVar("M", bound=FieldMarker)


@dataclasses.dataclass(frozen=True)
class Marked(Generic[M]):
    """A field that holds a marker, as :meth:`ConfigModel.find` returns it."""

    path: str
    """The field's full dotted path; list and tuple entries are indexed (``a.0.b``)."""
    marker: M
    """The marker on the field."""
    value: Any
    """The field's current value."""


def Field(default: Any = ..., *markers: FieldMarker, **kwargs: Any) -> Any:  # noqa: N802
    """Pydantic's ``Field``, with markers attached.

    Args:
        default: The field's default; ``...`` for a required field.
        markers: Markers to attach to the field.
        kwargs: Any other argument of :func:`pydantic.Field`.

    Raises:
        TypeError: If a marker is not a :class:`FieldMarker`.
    """
    for marker in markers:
        if not isinstance(marker, FieldMarker):
            raise TypeError(
                f"Field markers must be FieldMarker instances; got {marker!r}."
            )
    info = PydanticField(default, **kwargs)
    info.metadata.extend(markers)
    return info


def markers_of(info: FieldInfo) -> list[FieldMarker]:
    """Return the markers on a field, in the order they were attached."""
    return [meta for meta in info.metadata if isinstance(meta, FieldMarker)]


def find_markers(model: BaseModel, kind: type[M]) -> list[Marked[M]]:
    """Return every field in ``model``'s tree that holds a ``kind`` marker.

    Descends into nested models, including those held by tagged unions, lists,
    tuples and dicts. A field with several ``kind`` markers appears once per marker.
    """
    return list(_iter_marked(model, "", kind))


def _iter_marked(node: BaseModel, prefix: str, kind: type[M]) -> Iterator[Marked[M]]:
    for name, info in type(node).model_fields.items():
        path = f"{prefix}{name}"
        value = getattr(node, name)
        for marker in markers_of(info):
            if isinstance(marker, kind):
                yield Marked(path, marker, value)
        for child_path, child in _nested_models(value, path):
            yield from _iter_marked(child, f"{child_path}.", kind)


def _nested_models(value: Any, path: str) -> Iterator[tuple[str, BaseModel]]:
    if isinstance(value, BaseModel):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _nested_models(item, f"{path}.{index}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _nested_models(item, f"{path}.{key}")


def with_updates(data: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return a dumped config with ``updates`` set at their full dotted paths.

    Tuples in ``data`` come back as lists, so their entries can be set; Pydantic
    validates a list into a tuple field. A segment indexes a list when the container
    at that point is a list, and is a key otherwise.

    Raises:
        ValueError: If a path does not name a field of the dumped config.
    """
    thawed: dict[str, Any] = _thaw(data)
    for path, value in updates.items():
        *parents, last = path.split(".")
        cursor: Any = thawed
        for segment in parents:
            cursor = cursor[_key(cursor, segment, path)]
        cursor[_key(cursor, last, path)] = value
    return thawed


def _thaw(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _key(container: Any, segment: str, path: str) -> Any:
    if isinstance(container, list):
        if segment.isdigit() and int(segment) < len(container):
            return int(segment)
    elif isinstance(container, dict) and segment in container:
        return segment
    raise ValueError(f"{path!r} is not a path in this config.")


class Shown:
    """A value rendered by a marker: its ``repr`` is the marker's text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def marker_text(markers: list[FieldMarker], hook: str, arg: Any) -> str | None:
    """The first text a marker's ``hook`` returns for ``arg``, or ``None``."""
    for marker in markers:
        text = getattr(marker, hook)(arg)
        if text is not None:
            return str(text)
    return None


def value_shower(model: BaseModel) -> Callable[[str, Any], Any]:
    """Return ``show(path, value)``: ``value`` as the markers on ``model``'s field at
    ``path`` render it, or ``value`` itself.

    A rendered value comes back as :class:`Shown`, whose ``repr`` is the marker's
    text, so it drops into any printer that writes ``repr(value)``. Paths are full
    dotted paths, as :meth:`ConfigModel.to_flat_dict` gives; :data:`MISSING` and
    paths that are not marked fields are left as they are.
    """
    markers: dict[str, list[FieldMarker]] = {}
    for found in find_markers(model, FieldMarker):
        markers.setdefault(found.path, []).append(found.marker)

    def show(path: str, value: Any) -> Any:
        if path not in markers or value is MISSING:
            return value
        text = marker_text(markers[path], "format_value", value)
        return value if text is None else Shown(text)

    return show
