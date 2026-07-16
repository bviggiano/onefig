from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from onefig import ConfigError, ConfigModel, format_validation_error, tagged_union


@dataclass
class Adam:
    name: Literal["adam"] = "adam"
    lr: float = 1e-3

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be > 0")


@dataclass
class SGD:
    name: Literal["sgd"] = "sgd"
    lr: float = 1e-2


Optimizer = tagged_union(Adam, SGD)


class Inner(ConfigModel):
    depth: int = 1


class Cfg(ConfigModel):
    optimizer: Optimizer = Adam()
    epochs: int = 10
    inner: Inner = Inner()
    required: str  # no default


def _error(data: dict) -> ValidationError:
    try:
        Cfg.model_validate(data)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def _valid() -> dict:
    return {"required": "x"}


# ---- ConfigError -----------------------------------------------------------


def test_config_error_is_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


def test_urls_are_never_present() -> None:
    out = format_validation_error(_error({"epochs": "abc"}), root=Cfg, color=False)
    assert "errors.pydantic.dev" not in out


# ---- tree structure --------------------------------------------------------


def test_output_is_a_tree_with_header() -> None:
    out = format_validation_error(
        _error({"epochs": "abc", "required": "x"}), root=Cfg, name="run", color=False
    )
    assert out.splitlines()[0] == "✗ Config Validation Error in 'run' — 1 problem found"
    assert "└── epochs" in out
    assert "✗" in out


def test_header_pluralizes_and_counts() -> None:
    out = format_validation_error(
        _error({"epochs": "abc"}), root=Cfg, name="run", color=False
    )
    # missing `required` + bad `epochs` == 2 problems
    expected = "✗ Config Validation Error in 'run' — 2 problems found"
    assert out.splitlines()[0] == expected


def test_nested_paths_nest_in_the_tree() -> None:
    out = format_validation_error(
        _error({**_valid(), "inner": {"depth": "deep"}}), root=Cfg, color=False
    )
    assert "├── inner" in out or "└── inner" in out
    assert "depth" in out


def test_header_falls_back_to_class_name() -> None:
    out = format_validation_error(_error({"epochs": "abc"}), root=Cfg, color=False)
    assert out.splitlines()[0].startswith("✗ Config Validation Error in 'Cfg' — ")


# ---- message normalization -------------------------------------------------


def test_missing_message() -> None:
    out = format_validation_error(_error({}), root=Cfg, color=False)
    assert "required  ✗ required field is missing" in out


def test_extra_field_message() -> None:
    out = format_validation_error(
        _error({**_valid(), "nope": 1}), root=Cfg, color=False
    )
    assert "nope  ✗ unknown field" in out


def test_parsing_message_includes_input() -> None:
    out = format_validation_error(
        _error({**_valid(), "epochs": "abc"}), root=Cfg, color=False
    )
    assert "must be a valid integer (got 'abc')" in out


def test_value_error_prefix_is_stripped() -> None:
    out = format_validation_error(
        _error({**_valid(), "optimizer": {"name": "adam", "lr": -1.0}}),
        root=Cfg,
        color=False,
    )
    assert "must be > 0" in out
    assert "Value error," not in out


def test_union_tag_folds_into_field_label() -> None:
    # `optimizer` IS the adam variant; the tag is a label, not a nesting level.
    out = format_validation_error(
        _error({**_valid(), "optimizer": {"name": "adam", "lr": -1.0}}),
        root=Cfg,
        color=False,
    )
    assert "optimizer (adam)" in out
    assert "adam\n" not in out  # tag is never its own node


def test_drills_to_the_named_leaf_field() -> None:
    # The post_init message names `lr`; the branch reaches an `lr` leaf and the
    # message no longer repeats the field name.
    out = format_validation_error(
        _error({**_valid(), "optimizer": {"name": "adam", "lr": -1.0}}),
        root=Cfg,
        color=False,
    )
    assert "└── lr  ✗ must be > 0" in out


def test_unknown_tag_message() -> None:
    out = format_validation_error(
        _error({**_valid(), "optimizer": {"name": "rmsprop"}}), root=Cfg, color=False
    )
    assert "'rmsprop' is not a valid name" in out
    assert "adam" in out and "sgd" in out


# ---- source locations ------------------------------------------------------


def test_location_points_to_this_file() -> None:
    out = format_validation_error(
        _error({**_valid(), "epochs": "abc"}), root=Cfg, color=False
    )
    assert "→ " in out
    assert "test_errors.py:" in out


def test_location_resolves_union_variant() -> None:
    # The bound lives in Adam.__post_init__; the pointer should land on Adam.
    out = format_validation_error(
        _error({**_valid(), "optimizer": {"name": "adam", "lr": -1.0}}),
        root=Cfg,
        color=False,
    )
    line = next(ln for ln in out.splitlines() if "→ " in ln)
    assert "test_errors.py:" in line


def test_no_root_omits_locations() -> None:
    out = format_validation_error(_error({"epochs": "abc"}), color=False)
    assert "→ " not in out
    assert "epochs" in out  # still renders the tree and message


# ---- color -----------------------------------------------------------------


def test_color_emits_ansi() -> None:
    out = format_validation_error(_error({"epochs": "abc"}), root=Cfg, color=True)
    assert "\033[" in out


def test_no_color_is_plain() -> None:
    out = format_validation_error(_error({"epochs": "abc"}), root=Cfg, color=False)
    assert "\033[" not in out


# ---- integration through the public entry points ---------------------------


def test_load_raises_config_error_tree(tmp_path: Path) -> None:
    p = tmp_path / "run.yaml"
    p.write_text("epochs: not-an-int\nrequired: x\n")
    with pytest.raises(ConfigError) as info:
        Cfg.load(p)
    text = str(info.value)
    expected = "✗ Config Validation Error in 'run' — 1 problem found"
    assert text.splitlines()[0] == expected
    assert "errors.pydantic.dev" not in text


def test_update_from_cli_raises_config_error() -> None:
    cfg = Cfg(required="x")
    with pytest.raises(ConfigError) as info:
        cfg.update_from_cli(["epochs=not-an-int"])
    assert "must be a valid integer" in str(info.value)


def test_unknown_key_still_raises_plain_value_error() -> None:
    cfg = Cfg(required="x")
    with pytest.raises(ValueError, match="not found"):
        cfg.update_from_cli(["nope=1"])
