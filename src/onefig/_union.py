"""Discriminated ("tagged") unions over a set of config variants.

A tagged union lets one field accept any of several variant types, chosen by a
shared *tag* field that every variant carries as a distinct ``Literal``. YAML/CLI
then select the variant by its tag value, and Pydantic validates the payload
against exactly that variant's fields. :func:`tagged_union` builds the annotation
so a config can declare the field without hand-writing the
``Annotated[Union[...], Field(discriminator=...)]`` boilerplate.
"""

from __future__ import annotations

from typing import Annotated, Any, Union

from pydantic import Field


def tagged_union(*variants: type, tag: str = "name") -> Any:
    """Build a discriminated-union annotation over ``variants``, keyed by ``tag``.

    Each variant must carry ``tag`` as a distinct ``Literal`` field (the
    discriminator) — e.g. ``name: Literal["local"]`` on one variant and
    ``name: Literal["s3"]`` on another. Assign the result to a module-level alias
    and use it as a field type; a value's ``tag`` picks the variant, and
    validation runs against that variant alone::

        from typing import Literal
        from pydantic import BaseModel
        from onefig import ConfigModel, tagged_union

        class Local(BaseModel):
            backend: Literal["local"] = "local"
            path: str = "./data"

        class S3(BaseModel):
            backend: Literal["s3"] = "s3"
            bucket: str

        Store = tagged_union(Local, S3, tag="backend")

        class AppConfig(ConfigModel):
            store: Store = Local()

    Variants may be Pydantic models or stdlib dataclasses (Pydantic validates
    both), so a schema defined elsewhere as plain dataclasses can be composed
    here without redefining it.

    A type checker cannot follow a union built at runtime, so a field annotated
    with the returned alias is seen as ``Any``. In a type-checked codebase, guard
    the alias so the checker still sees a real union (and can narrow it after an
    ``isinstance`` check) while the runtime keeps the discriminator::

        from typing import TYPE_CHECKING, Union

        if TYPE_CHECKING:
            Store = Union[Local, S3]
        else:
            Store = tagged_union(Local, S3, tag="backend")

    Args:
        variants: Two or more variant types sharing a ``tag`` discriminator field.
        tag: Name of the discriminator field common to every variant (default
            ``"name"``).

    Returns:
        An ``Annotated[Union[...], Field(discriminator=tag)]`` type to use as a
        field annotation.

    Raises:
        ValueError: If fewer than two variants are given (a union needs at least
            two members).
    """
    if len(variants) < 2:
        raise ValueError(
            f"tagged_union needs at least two variants, got {len(variants)}."
        )
    return Annotated[Union[variants], Field(discriminator=tag)]  # type: ignore[valid-type]
