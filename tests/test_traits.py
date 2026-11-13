from __future__ import annotations

import dataclasses
from typing import Annotated, Optional

import pytest
from pydantic import model_validator
from typing_extensions import Self

from onefig import ConfigError, ConfigModel, Field, FieldTrait, tagged_union
from onefig._help import format_help


@dataclasses.dataclass(frozen=True)
class Range(FieldTrait):
    low: float
    high: float

    def describe(self) -> str:
        return f"Drawn from {self.low:g} to {self.high:g}."


@dataclasses.dataclass(frozen=True)
class LogRange(Range):
    pass


@dataclasses.dataclass(frozen=True)
class Unit(FieldTrait):
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


def test_find_returns_every_trait_of_a_kind_with_its_path_and_value() -> None:
    found = World().find(Range)
    assert [(m.path, m.value) for m in found] == [
        ("diseases", 2),
        ("cohorts.0.size", None),
        ("cohorts.1.size", 900),
        ("by_site.north.size", None),
        ("waves.0.size", None),
        ("store.rate", 0.1),
    ]
    assert isinstance(found[1].trait, LogRange)


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


def test_traits_survive_subclassing_the_config() -> None:
    class Bigger(World):
        extra: int = Field(0, Range(0, 2))

    assert [m.path for m in Bigger().find(Range)][-1] == "extra"


def test_field_keeps_pydantic_arguments() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        World(weight=-1)


def test_field_rejects_objects_that_are_not_traits() -> None:
    with pytest.raises(TypeError, match="must be FieldTrait instances"):
        Field(1, Range(0, 1), "not a trait")  # type: ignore[arg-type]


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


def test_help_shows_what_a_trait_describes() -> None:
    out = format_help(World())
    assert "Drawn from 1 to 8." in out
    assert "Drawn from 0 to 1." in out  # inside the tagged union


# ---- how fields with traits are printed ----------------------------------------


@dataclasses.dataclass(frozen=True)
class Drawn(FieldTrait):
    """Shows an unset value as drawn, and the type without ``None``."""

    low: int
    high: int

    def format_value(self, value: object) -> str | None:
        return f"<drawn {self.low}-{self.high}>" if value is None else None

    def format_type(self, annotation: object) -> str:
        return "int (drawn)"


@dataclasses.dataclass(frozen=True)
class Silent(FieldTrait):
    """Leaves every rendering to onefig, or to the next trait."""


class Study(ConfigModel):
    size: Optional[int] = Field(None, Silent(), Drawn(10, 99))  # noqa: UP045


class Run(ConfigModel):
    name: str
    size: Optional[int] = Field(None, Drawn(1, 8))  # noqa: UP045
    studies: list[Study] = [Study(), Study(size=50)]


def test_help_shows_a_field_as_its_traits_render_it() -> None:
    out = Run(name="r", size=3).format_help()
    assert "size : int (drawn) = 3" in out
    assert "(default: <drawn 1-8>)" in out


def test_the_first_trait_that_renders_wins() -> None:
    assert "<drawn 10-99>" in Study().format_help()


def test_help_on_the_class_shows_defaults_and_required_fields() -> None:
    out = Run.format_help()
    assert "name : str = <required>" in out
    assert "size : int (drawn) = <drawn 1-8>" in out
    assert Study.format_help() == Study().format_help()


def test_display_renders_trait_leaves_in_nested_lists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    Run(name="r").display()
    out = capsys.readouterr().out
    assert "size: <drawn 1-8>" in out
    assert "size: <drawn 10-99>" in out  # studies.0.size
    assert "size: 50" in out  # set, so shown as is


def test_diff_from_defaults_compares_values_before_rendering_them() -> None:
    lines = Run(name="r", size=3).format_diff_from_defaults(color=False).splitlines()
    assert any(line.split() == ["size", "<drawn", "1-8>", "→", "3"] for line in lines)
    # Unchanged and unset: shown once as drawn, not as an override.
    assert any(line.split() == ["studies.0.size", "<drawn", "10-99>"] for line in lines)


def test_diff_renders_both_sides() -> None:
    out = Run(name="r").format_diff(Run(name="r", size=4), color=False)
    assert out.split() == ["size", "<drawn", "1-8>", "→", "4"]
