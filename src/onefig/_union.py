"""Tagged unions over a set of config variants.

A tagged union lets one field accept any of several variant types. In YAML, the
CLI and env vars, a *tag* key (``name`` by default) picks the variant, and
Pydantic validates the remaining keys against exactly that variant's fields.

The tag belongs to the variant *class*, not its instances: it is a ``ClassVar``
(set explicitly, or derived from the class name), so it never shows up as a
setting in ``model_fields``, ``--help`` or a constructor call. It is still written
into every dump (``to_dict``, ``to_flat_dict``, ``save_yaml``, ``diff``), so a
saved config records which variant it used and loads back to the same one.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Annotated, Any, Callable, Union, get_args, get_origin

from pydantic import BaseModel, BeforeValidator, Discriminator, Tag, WrapSerializer
from pydantic.fields import FieldInfo

try:
    from types import UnionType  # Python 3.10+
except ImportError:  # pragma: no cover
    UnionType = None  # type: ignore[assignment,misc]

# A word starts at a capital that follows a lowercase letter or digit
# (``cosineWarmup``), or at the last capital of an acronym run when a lowercase
# letter follows it (``UKB|Fields``).
_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_CONFIG_SUFFIX = "Config"


@dataclasses.dataclass(frozen=True, eq=False)
class UnionInfo:
    """Marker attached to a :func:`tagged_union` annotation.

    onefig's override engine, help and error formatting read it to find the
    tag key and the variant class behind each tag value.
    """

    tag: str
    variants: Mapping[str, type]

    def tag_of(self, value: Any) -> str | None:
        """Return the tag of a variant instance, or ``None`` if it is not one."""
        for cls in type(value).__mro__:
            for tag_value, variant in self.variants.items():
                if variant is cls:
                    return tag_value
        return None

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Any) -> Any:
        schema = handler(core_schema)
        refs = [
            choice["allOf"][0].get("$ref")
            for choice in schema.get("oneOf", [])
            if "allOf" in choice
        ]
        if len(refs) == len(self.variants) and all(refs):
            schema["discriminator"] = {
                "propertyName": self.tag,
                "mapping": dict(zip(self.variants, refs)),
            }
        return schema


@dataclasses.dataclass(frozen=True)
class _TagJsonSchema:
    """Adds the tag as a required ``const`` property to a variant's JSON schema."""

    tag: str
    value: str

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Any) -> Any:
        return {
            "allOf": [
                handler(core_schema),
                {
                    "properties": {self.tag: {"const": self.value}},
                    "required": [self.tag],
                },
            ]
        }


def derive_tag(class_name: str) -> str:
    """Derive a variant's tag from its class name.

    Leading underscores and a trailing ``Config`` are dropped (unless nothing
    would be left), then the CamelCase name becomes snake_case, with an acronym
    run kept together as one word: ``MCARConfig`` → ``mcar``,
    ``CosineWarmupConfig`` → ``cosine_warmup``, ``UKBFieldsConfig`` →
    ``ukb_fields``, ``AdamW`` → ``adam_w``.

    Args:
        class_name: The variant's class name.

    Returns:
        The derived tag.
    """
    name = class_name.lstrip("_") or class_name
    if name.endswith(_CONFIG_SUFFIX) and name != _CONFIG_SUFFIX:
        name = name[: -len(_CONFIG_SUFFIX)]
    return _WORD_BOUNDARY.sub("_", name).lower()


def tagged_union(*variants: type, tag: str = "name") -> Any:
    """Build a tagged-union annotation over ``variants``, selected by ``tag``.

    Each variant is a Pydantic model (preferably a ``ConfigModel``) or a stdlib
    dataclass whose fields are only its real settings. Its tag is a class
    attribute: an explicit ``name: ClassVar[str] = "..."`` when set on the class
    itself, otherwise derived from the class name by :func:`derive_tag` and set
    on the class, so ``cfg.store.name`` reads it either way::

        from typing import ClassVar
        from onefig import ConfigModel, tagged_union

        class LocalConfig(ConfigModel):  # tag derived: "local"
            path: str = "./data"

        class S3Store(ConfigModel):
            name: ClassVar[str] = "s3"  # explicit: wins over derived "s3_store"
            bucket: str

        Store = tagged_union(LocalConfig, S3Store)

        class AppConfig(ConfigModel):
            store: Store = LocalConfig()

    YAML selects a variant with ``store: {name: s3, bucket: b}``, and the CLI
    switches it with ``store.name=s3 store.bucket=b``.

    A type checker cannot follow a union built at runtime, so it sees a field
    annotated with the returned alias as ``Any``. In a type-checked codebase,
    guard the alias so the checker still sees a real union::

        from typing import TYPE_CHECKING, Union

        if TYPE_CHECKING:
            Store = Union[LocalConfig, S3Store]
        else:
            Store = tagged_union(LocalConfig, S3Store)

    Args:
        variants: Two or more variant classes.
        tag: Name of the tag key in YAML/CLI/env and of the class attribute that
            holds each variant's tag (default ``"name"``).

    Returns:
        An ``Annotated`` union type to use as a field annotation.

    Raises:
        ValueError: If fewer than two variants are given, or two variants share
            a tag.
        TypeError: If a variant is not a Pydantic model or dataclass, declares
            the tag as a field, or sets it to something other than a non-empty
            string.
    """
    if len(variants) < 2:
        raise ValueError(
            f"tagged_union needs at least two variants, got {len(variants)}."
        )
    by_tag: dict[str, type] = {}
    for cls in variants:
        value = _variant_tag(cls, tag)
        if value in by_tag:
            raise ValueError(
                f"tagged_union variants {by_tag[value].__name__} and {cls.__name__} "
                f"share the {tag} {value!r}; set an explicit "
                f"`{tag}: ClassVar[str]` on one of them."
            )
        by_tag[value] = cls
    for value, cls in by_tag.items():
        if tag not in vars(cls):
            setattr(cls, tag, value)

    info = UnionInfo(tag=tag, variants=by_tag)
    choices = tuple(
        Annotated[
            cls,
            BeforeValidator(_tag_stripper(tag)),
            _TagJsonSchema(tag, value),
            Tag(value),  # must be the outermost annotation of each choice
        ]
        for value, cls in by_tag.items()
    )
    return Annotated[  # type: ignore[valid-type]
        Union[choices],
        Discriminator(_selector(info)),
        WrapSerializer(_tag_writer(info)),
        info,
    ]


def union_info(annotation: Any) -> UnionInfo | None:
    """Return the :class:`UnionInfo` of a tagged-union field, or ``None``.

    Accepts a ``FieldInfo`` or an annotation, and looks through ``Optional`` /
    ``X | None`` so an optional tagged union is found too.
    """
    if isinstance(annotation, FieldInfo):
        for meta in annotation.metadata:
            if isinstance(meta, UnionInfo):
                return meta
        annotation = annotation.annotation
    origin = get_origin(annotation)
    if origin is Annotated:
        for meta in annotation.__metadata__:
            if isinstance(meta, UnionInfo):
                return meta
        return None
    if origin is Union or (UnionType is not None and origin is UnionType):
        for arg in get_args(annotation):
            found = union_info(arg)
            if found is not None:
                return found
    return None


def _variant_tag(cls: Any, tag: str) -> str:
    """Return the explicit tag set on ``cls`` itself, else one derived from its name."""
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        field_names = set(cls.model_fields)
    elif isinstance(cls, type) and dataclasses.is_dataclass(cls):
        field_names = {f.name for f in dataclasses.fields(cls)}
    else:
        raise TypeError(
            f"tagged_union variants must be Pydantic models or dataclasses; "
            f"got {cls!r}."
        )
    if tag in field_names:
        raise TypeError(
            f"{cls.__name__}.{tag} is a field; remove it (the tag is derived from "
            f'the class name) or declare it as `{tag}: ClassVar[str] = "..."`. '
            "The tag selects the variant and is not a setting."
        )
    if tag in vars(cls):  # set on this class, not inherited from a parent variant
        value = vars(cls)[tag]
        if not isinstance(value, str) or not value:
            raise TypeError(
                f"{cls.__name__}.{tag} must be a non-empty string tag; got {value!r}."
            )
        return value
    return derive_tag(cls.__name__)


def _selector(info: UnionInfo) -> Callable[[Any], str | None]:
    """Build the discriminator: read the tag from raw input or a variant instance."""

    def select(value: Any) -> str | None:
        if isinstance(value, dict):
            return value.get(info.tag)
        return info.tag_of(value)

    # Pydantic names the discriminator after this function in its error
    # context ("name()"); onefig's error formatting strips the parentheses.
    select.__name__ = info.tag
    return select


def _tag_stripper(tag: str) -> Callable[[Any], Any]:
    """Drop the tag key from raw input so the variant validates only its fields."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if k != tag}
        return value

    return strip


def _tag_writer(info: UnionInfo) -> Callable[[Any, Any], Any]:
    """Write the variant's tag back in front of its serialized fields."""

    def write(instance: Any, handler: Any) -> Any:
        return {info.tag: info.tag_of(instance), **handler(instance)}

    return write
