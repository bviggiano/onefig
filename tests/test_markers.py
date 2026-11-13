from __future__ import annotations

import dataclasses
from typing import Annotated, Optional

import pytest
from pydantic import model_validator
from typing_extensions import Self

from onefig import ConfigError, ConfigModel, Field, FieldMarker, tagged_union
from onefig._help import format_help


@dataclasses.dataclass(frozen=True)
class Range(FieldMarker):
    low: float
    high: float

    def describe(self) -> str:
        return f"Drawn from {self.low:g} to {self.high:g}."


@dataclasses.dataclass(frozen=True)
class LogRange(Range):
    pass


@dataclasses.dataclass(frozen=True)
class Unit(FieldMarker):
    name: str


class Cohort(ConfigModel):
    size: Optional[int] = Field(None, LogRange(500, 150_000))  # noqa: UP045
    label: str = "a"


class Local(ConfigModel):
    rate: float = Field(0.1, Range(0, 1), Unit("1/s"))


class Remote(ConfigModel):
    retries: int = Field(3, Range(0, 5))


class World(ConfigModel):
    diseases: Annotated[int, Range(1, 8)] = 2
    weight: float = Field(1.0, Unit("kg"), ge=0)
    cohorts: list[Cohort] = [Cohort(), Cohort(size=900)]
    by_site: dict[str, Cohort] = {"north": Cohort()}
    waves: tuple[Cohort, ...] = (Cohort(),)
    store: tagged_union(Local, Remote) = Local()  # type: ignore[valid-type]

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.diseases > 6 and self.weight < 1:
            raise ValueError("many diseases need weight of at least 1")
        return self


def test_find_returns_every_marker_of_a_kind_with_its_path_and_value() -> None:
    found = World().find(Range)
    assert [(m.path, m.value) for m in found] == [
        ("diseases", 2),
        ("cohorts.0.size", None),
        ("cohorts.1.size", 900),
        ("by_site.north.size", None),
        ("waves.0.size", None),
        ("store.rate", 0.1),
    ]
    assert isinstance(found[1].marker, LogRange)


def test_find_follows_the_current_variant_of_a_tagged_union() -> None:
    world = World(store=Remote())
    assert [m.path for m in world.find(Range)][-1] == "store.retries"


def test_find_matches_only_the_kind_asked_for() -> None:
    assert [m.path for m in World().find(Unit)] == ["weight", "store.rate"]
    assert [m.path for m in World().find(LogRange)] == [
        "cohorts.0.size",
        "cohorts.1.size",
        "by_site.north.size",
        "waves.0.size",
    ]


def test_markers_survive_subclassing_the_config() -> None:
    class Bigger(World):
        extra: int = Field(0, Range(0, 2))

    assert [m.path for m in Bigger().find(Range)][-1] == "extra"


def test_field_keeps_pydantic_arguments() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        World(weight=-1)


def test_field_rejects_objects_that_are_not_markers() -> None:
    with pytest.raises(TypeError, match="must be FieldMarker instances"):
        Field(1, Range(0, 1), "not a marker")  # type: ignore[arg-type]


def test_replace_accepts_the_paths_find_returns() -> None:
    world = World()
    drawn = {m.path: 7 for m in world.find(Range) if m.value is None}
    replaced = world.replace(drawn)
    assert [m.value for m in replaced.find(LogRange)] == [7, 900, 7, 7]
    assert [m.value for m in world.find(LogRange)] == [None, 900, None, None]


def test_replace_works_on_a_frozen_config_and_returns_an_unfrozen_copy() -> None:
    world = World().freeze()
    replaced = world.replace({"diseases": 3})
    assert replaced.diseases == 3
    assert world.diseases == 2
    assert not replaced.is_frozen


def test_replace_keeps_the_config_name() -> None:
    world = World()
    world.config_name = "baseline"
    assert world.replace({"diseases": 3}).config_name == "baseline"


def test_replace_validates_all_updates_together() -> None:
    world = World()
    # Applied one at a time, diseases=7 would fail before weight=2 lands.
    assert world.replace({"diseases": 7, "weight": 2}).diseases == 7
    with pytest.raises(ConfigError):
        world.replace({"diseases": 7, "weight": 0.5})


@pytest.mark.parametrize(
    "path", ["missing", "cohorts.5.size", "cohorts.x", "waves.1.size", "weight.x"]
)
def test_replace_rejects_paths_outside_the_config(path: str) -> None:
    with pytest.raises(ValueError, match="is not a path in this config"):
        World().replace({path: 1})


def test_help_shows_what_a_marker_describes() -> None:
    out = format_help(World())
    assert "Drawn from 1 to 8." in out
    assert "Drawn from 0 to 1." in out  # inside the tagged union
