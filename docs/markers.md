# Field markers

A field marker is a typed object you attach to a config field to say something
onefig itself does not interpret: a range to draw the value from, a unit, a flag
for values that must not be logged. onefig carries the marker on the field, finds
every field holding a given kind of marker anywhere in the config tree, and shows
what the marker describes in `--help`. What a marker means is up to your code.

## Define a marker

Subclass `FieldMarker`, usually as a frozen dataclass:

```python
import dataclasses

from onefig import FieldMarker


@dataclasses.dataclass(frozen=True)
class Range(FieldMarker):
    low: float
    high: float

    def describe(self) -> str:
        return f"Drawn from {self.low:g} to {self.high:g} when unset."
```

`describe()` is optional. Its text is shown after the field's description in
`--help`; returning `None` (the default) shows nothing. Two more hooks control how
marked fields are printed; see [How marked fields are printed](#how-marked-fields-are-printed).

## Attach it

`onefig.Field` is Pydantic's `Field` with markers after the default. Every other
argument passes through to Pydantic:

```python
from typing import Optional

from onefig import ConfigModel, Field


class CohortConfig(ConfigModel):
    size: Optional[int] = Field(None, Range(500, 150_000), ge=500, le=150_000)
    """People in the cohort."""


class WorldConfig(ConfigModel):
    diseases: Optional[int] = Field(None, Range(1, 8))
    cohorts: list[CohortConfig] = [CohortConfig(), CohortConfig(size=900)]
```

A marker can also be attached through `Annotated`, as in
`diseases: Annotated[int, Range(1, 8)] = 2`. A field can hold several markers, and
markers are inherited by subclasses of the config.

!!! tip "Why not subclass `FieldInfo`?"
    Pydantic merges field information from `Annotated` and the default into a
    plain `FieldInfo`, so a subclass and its extra attributes can be dropped
    without an error. Markers live in `FieldInfo.metadata`, which survives every
    way of declaring a field.

## Find marked fields

`cfg.find(Kind)` returns every field holding a `Kind` marker (or a subclass of
it), in declaration order, as `Marked(path, marker, value)`:

```python
cfg = WorldConfig()
for m in cfg.find(Range):
    print(m.path, m.marker, m.value)
# diseases Range(low=1, high=8) None
# cohorts.0.size Range(low=500, high=150000) None
# cohorts.1.size Range(low=500, high=150000) 900
```

The walk descends into nested configs, the current variant of a
[tagged union](unions.md), and configs held in lists, tuples and dicts. Paths use
the same dotted form as `to_flat_dict`, with list entries indexed.

## Fill them in

`cfg.replace(updates)` returns a new config with the updates applied and the
whole config validated once, so cross-field validators see the final values. It
takes the full dotted paths `find` returns, leaves the original unchanged (so it
works on a frozen config), and keeps `config_name` and `commit_hash`:

```python
import random

rng = random.Random(0)
drawn = {
    m.path: rng.randint(int(m.marker.low), int(m.marker.high))
    for m in cfg.find(Range)
    if m.value is None
}
resolved = cfg.replace(drawn).freeze()
```

An unknown path raises `ValueError`; a value that fails validation raises
`ConfigError`.

## How marked fields are printed

A marker decides how its fields look wherever a config is printed, through three
optional hooks. The hooks live on the marker class, so a config class only declares
its fields, and a helper that builds fields with a marker (a `KnobField(...)` that
returns `onefig.Field(None, Knob(...))`, say) gets its printing from that marker.

| Hook | Used for | Shown in |
|---|---|---|
| `describe()` | a note after the field's description | `--help` |
| `format_value(value)` | the field's value | the current and default value in `--help`, the field's leaf in `display()` and `--show`, both sides of `format_diff` and `format_diff_from_defaults` |
| `format_type(annotation)` | the field's type | `--help` |

Each returns a string, or `None` to keep onefig's own rendering, so a marker
overrides only what it needs. When a field has several markers, the first that
returns text wins; every `describe()` text is shown. Values are compared before
they are rendered, so a diff never reports a field as changed because of how it is
shown.

```python
@dataclasses.dataclass(frozen=True)
class Range(FieldMarker):
    low: int
    high: int

    def describe(self) -> str:
        return f"Drawn from {self.low} to {self.high} when unset."

    def format_value(self, value: object) -> str | None:
        return f"<drawn {self.low}-{self.high}>" if value is None else None

    def format_type(self, annotation: object) -> str:
        return "int"


class WorldConfig(ConfigModel):
    diseases: Optional[int] = Field(None, Range(1, 8))
    """Diseases in the world."""


WorldConfig.print_help()
```

```text
│   diseases : int = <drawn 1-8>                                                       │
│       Diseases in the world. Drawn from 1 to 8 when unset.                           │
```

`print_help()` and `format_help()` also work on the class, as above: they show its
defaults, and a required field as `<required>`.
