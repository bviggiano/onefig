"""Field traits: typed objects a project attaches to config fields.

onefig does not interpret a trait. It carries the trait on the field, finds every
field holding a given kind of trait with :meth:`ConfigModel.find`, and asks the
trait how the field should look wherever a config is printed. What a trait means,
such as a range to draw the field's value from, is up to the project that defines
it, and so is how its fields are shown: that lives on the trait class, never on the
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


class FieldTrait:
    """Base class for traits attached to config fields with :func:`Field`.

    Subclasses are usually frozen dataclasses. A trait can also be attached
    through ``Annotated[int, MyTrait(...)]``.

    Three hooks set how a field with a trait is printed. Each returns ``None`` to leave
    onefig's own rendering, so a trait overrides only what it needs; when a field
    has several traits, the first that returns text wins (``describe`` texts are
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


T = TypeVar("T", bound=FieldTrait)


@dataclasses.dataclass(frozen=True)
class FoundTrait(Generic[T]):
    """A field that holds a trait, as :meth:`ConfigModel.find` returns it."""

    path: str
    """The field's full dotted path; list and tuple entries are indexed (``a.0.b``)."""
    trait: T
    """The trait on the field."""
    value: Any
    """The field's current value."""


def Field(default: Any = ..., *traits: FieldTrait, **kwargs: Any) -> Any:  # noqa: N802
    """Pydantic's ``Field``, with traits attached.

    Args:
        default: The field's default; ``...`` for a required field.
        traits: Traits to attach to the field.
        kwargs: Any other argument of :func:`pydantic.Field`.

    Raises:
        TypeError: If a trait is not a :class:`FieldTrait`.
    """
    for trait in traits:
        if not isinstance(trait, FieldTrait):
            raise TypeError(
                f"Field traits must be FieldTrait instances; got {trait!r}."
            )
    info = PydanticField(default, **kwargs)
    info.metadata.extend(traits)
    return info


def traits_of(info: FieldInfo) -> list[FieldTrait]:
    """Return the traits on a field, in the order they were attached."""
    return [meta for meta in info.metadata if isinstance(meta, FieldTrait)]


def find_traits(model: BaseModel, kind: type[T]) -> list[FoundTrait[T]]:
    """Return every field in ``model``'s tree that holds a ``kind`` trait.

    Descends into nested models, including those held by tagged unions, lists,
    tuples and dicts. A field with several ``kind`` traits appears once per trait.
    """
    return list(_iter_traits(model, "", kind))


def _iter_traits(
    node: BaseModel, prefix: str, kind: type[T]
) -> Iterator[FoundTrait[T]]:
    for name, info in type(node).model_fields.items():
        path = f"{prefix}{name}"
        value = getattr(node, name)
        for trait in traits_of(info):
            if isinstance(trait, kind):
                yield FoundTrait(path, trait, value)
        for child_path, child in _nested_models(value, path):
            yield from _iter_traits(child, f"{child_path}.", kind)


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
    """A value rendered by a trait: its ``repr`` is the trait's text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def trait_text(traits: list[FieldTrait], hook: str, arg: Any) -> str | None:
    """The first text a trait's ``hook`` returns for ``arg``, or ``None``."""
    for trait in traits:
        text = getattr(trait, hook)(arg)
        if text is not None:
            return str(text)
    return None


def value_shower(model: BaseModel) -> Callable[[str, Any], Any]:
    """Return ``show(path, value)``: ``value`` as the traits on ``model``'s field at
    ``path`` render it, or ``value`` itself.

    A rendered value comes back as :class:`Shown`, whose ``repr`` is the trait's
    text, so it drops into any printer that writes ``repr(value)``. Paths are full
    dotted paths, as :meth:`ConfigModel.to_flat_dict` gives; :data:`MISSING` and
    paths of fields without traits are left as they are.
    """
    traits: dict[str, list[FieldTrait]] = {}
    for found in find_traits(model, FieldTrait):
        traits.setdefault(found.path, []).append(found.trait)

    def show(path: str, value: Any) -> Any:
        if path not in traits or value is MISSING:
            return value
        text = trait_text(traits[path], "format_value", value)
        return value if text is None else Shown(text)

    return show
