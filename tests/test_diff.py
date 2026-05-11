from __future__ import annotations

import pytest

from onefig import MISSING, ConfigModel
from onefig._diff import (
    _MissingType,
    compute_diff,
    format_against_defaults,
    format_diff,
)


class Sub(ConfigModel):
    lr: float = 1e-4
    name: str = "default"


class Cfg(ConfigModel):
    epochs: int = 10
    debug: bool = False
    model: Sub = Sub()


class HasRequired(ConfigModel):
    name: str  # no default
    epochs: int = 10


# ---- MISSING sentinel ------------------------------------------------------


def test_missing_is_singleton() -> None:
    assert MISSING is _MissingType()
    assert MISSING is MISSING


def test_missing_is_falsy_with_distinctive_repr() -> None:
    assert bool(MISSING) is False
    assert "MISSING" in repr(MISSING)


# ---- compute_diff (low-level) ----------------------------------------------


def test_compute_diff_empty_when_equal() -> None:
    assert compute_diff({"a": 1, "b": 2}, {"a": 1, "b": 2}) == {}


def test_compute_diff_returns_old_new_tuples() -> None:
    assert compute_diff({"a": 1}, {"a": 2}) == {"a": (1, 2)}


def test_compute_diff_uses_missing_sentinel_for_added() -> None:
    out = compute_diff({"a": 1}, {"a": 1, "b": 2})
    assert out == {"b": (MISSING, 2)}


def test_compute_diff_uses_missing_sentinel_for_removed() -> None:
    out = compute_diff({"a": 1, "b": 2}, {"a": 1})
    assert out == {"b": (2, MISSING)}


def test_compute_diff_ordered_a_first_then_b_only() -> None:
    # `a` first (in its order), then `b`-only keys (in `b`'s order).
    a = {"x": 1, "y": 1, "z": 1}
    b = {"y": 2, "z": 1, "w": 99, "v": 88}
    out = compute_diff(a, b)
    assert list(out.keys()) == ["x", "y", "w", "v"]


# ---- ConfigModel.diff ------------------------------------------------------


def test_diff_against_configmodel() -> None:
    a = Cfg(epochs=10, model=Sub(lr=0.001, name="base"))
    b = Cfg(epochs=20, model=Sub(lr=0.001, name="finetune"))
    assert a.diff(b) == {
        "epochs": (10, 20),
        "model.name": ("base", "finetune"),
    }


def test_diff_empty_when_configs_equal() -> None:
    a = Cfg(epochs=5, model=Sub(lr=0.01, name="x"))
    b = Cfg(epochs=5, model=Sub(lr=0.01, name="x"))
    assert a.diff(b) == {}


def test_diff_against_nested_dict() -> None:
    a = Cfg(epochs=10, model=Sub(lr=0.001))
    # A nested dict that matches `a`'s full shape but with a different epochs.
    other = {
        "epochs": 20,
        "debug": False,
        "model": {"lr": 0.001, "name": "default"},
    }
    assert a.diff(other) == {"epochs": (10, 20)}


def test_diff_against_flat_dict() -> None:
    a = Cfg(epochs=10, model=Sub(lr=0.001, name="default"))
    other = {
        "epochs": 20,
        "debug": False,
        "model.lr": 0.001,
        "model.name": "default",
    }
    assert a.diff(other) == {"epochs": (10, 20)}


def test_diff_with_partial_dict_surfaces_removed_keys() -> None:
    # A partial dict (missing keys present in self) reports them as
    # (self_value, MISSING) — useful for confirming a logged-flat-dict
    # snapshot is complete.
    a = Cfg()
    partial = {"epochs": 10}
    out = a.diff(partial)
    assert ("debug",) == tuple(k for k in out if k == "debug")
    assert out["debug"] == (False, MISSING)


def test_diff_with_dict_surfaces_added_keys() -> None:
    # Cross-schema: dict has a key the config doesn't know about.
    a = Cfg()
    other = {**a.to_flat_dict(), "extra.field": 99}
    out = a.diff(other)
    assert out == {"extra.field": (MISSING, 99)}


def test_diff_rejects_unsupported_types() -> None:
    a = Cfg()
    with pytest.raises(TypeError, match="ConfigModel or dict"):
        a.diff(42)  # type: ignore[arg-type]


# ---- ConfigModel.diff_from_defaults ----------------------------------------


def test_diff_from_defaults_returns_only_overridden() -> None:
    cfg = Cfg(epochs=99, model=Sub(lr=0.5))
    out = cfg.diff_from_defaults()
    assert out == {
        "epochs": (10, 99),
        "model.lr": (1e-4, 0.5),
    }


def test_diff_from_defaults_empty_when_at_defaults() -> None:
    assert Cfg().diff_from_defaults() == {}


def test_diff_from_defaults_raises_when_required_fields_missing() -> None:
    cfg = HasRequired(name="explicit")
    with pytest.raises(ValueError, match="required fields"):
        cfg.diff_from_defaults()


# ---- format_diff -----------------------------------------------------------


def test_format_diff_empty_returns_message() -> None:
    assert format_diff({}, color=False) == "(no changes)"
    assert format_diff({}, color=False, empty_message="nope") == "nope"


def test_format_diff_basic_alignment() -> None:
    diff = {
        "epochs": (10, 20),
        "model.lr": (1e-4, 1e-3),
    }
    out = format_diff(diff, color=False)
    # Side-by-side: key  old  →  new, with old column padded.
    assert out == (
        "  epochs    10      →  20\n"
        "  model.lr  0.0001  →  0.001"
    )


def test_format_diff_renders_missing_on_added_side() -> None:
    out = format_diff({"experiment.id": (MISSING, "abc123")}, color=False)
    assert "<MISSING>" in out
    assert "→" in out
    assert "abc123" in out


def test_format_diff_renders_missing_on_removed_side() -> None:
    out = format_diff({"model.depth": (12, MISSING)}, color=False)
    assert "<MISSING>" in out
    assert "→" in out
    assert "12" in out


def test_format_diff_color_applies_ansi_codes() -> None:
    out = format_diff({"epochs": (10, 20)}, color=True)
    assert "\033[31m" in out   # red for old
    assert "\033[32m" in out   # green for new
    assert "\033[0m" in out    # reset


def test_format_diff_color_off_strips_ansi() -> None:
    out = format_diff({"epochs": (10, 20)}, color=False)
    assert "\033[" not in out


def test_format_diff_missing_is_dimmed_when_colored() -> None:
    out = format_diff({"k": (MISSING, "v")}, color=True)
    assert "\033[2m" in out   # dim for the MISSING placeholder


# ---- ConfigModel.format_diff / print_diff ----------------------------------


def test_format_diff_method_round_trips_via_diff() -> None:
    a = Cfg(epochs=10)
    b = Cfg(epochs=20)
    assert a.format_diff(b, color=False) == format_diff(
        a.diff(b), color=False
    )


def test_print_diff_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    a = Cfg(epochs=10)
    b = Cfg(epochs=20)
    a.print_diff(b, color=False)
    captured = capsys.readouterr()
    assert "epochs" in captured.out
    assert "10" in captured.out
    assert "20" in captured.out
    assert "→" in captured.out


# ---- format_against_defaults -----------------------------------------------


def test_format_against_defaults_shows_all_fields() -> None:
    out = format_against_defaults(
        current={"a": 1, "b": 2, "c": 3},
        defaults={"a": 1, "b": 2, "c": 3},
        color=False,
    )
    # Every field rendered, no overrides.
    assert "a" in out and "b" in out and "c" in out
    assert "→" not in out


def test_format_against_defaults_overridden_row_has_arrow() -> None:
    out = format_against_defaults(
        current={"epochs": 99, "debug": False},
        defaults={"epochs": 10, "debug": False},
        color=False,
    )
    # epochs is overridden (10 → 99); debug is unchanged.
    lines = out.split("\n")
    assert any("10" in ln and "99" in ln and "→" in ln for ln in lines)
    assert any("debug" in ln and "→" not in ln for ln in lines)


def test_format_against_defaults_unchanged_value_is_green() -> None:
    out = format_against_defaults(
        current={"a": 1},
        defaults={"a": 1},
        color=True,
    )
    # Unchanged value is colored green.
    assert "\033[32m" in out
    assert "\033[31m" not in out
    assert "→" not in out


def test_format_against_defaults_overridden_row_is_red_to_green() -> None:
    out = format_against_defaults(
        current={"a": 2},
        defaults={"a": 1},
        color=True,
    )
    assert "\033[31m" in out   # red for default
    assert "\033[32m" in out   # green for current
    assert "→" in out


def test_format_against_defaults_empty_returns_message() -> None:
    assert format_against_defaults({}, {}, color=False) == "(empty config)"


# ---- ConfigModel.format_diff_from_defaults ---------------------------------


def test_format_diff_from_defaults_includes_unchanged_fields() -> None:
    # Cfg has defaults epochs=10, debug=False, model.lr=1e-4, model.name="default".
    # Overriding only `epochs` should still surface every other field.
    cfg = Cfg(epochs=99)
    out = cfg.format_diff_from_defaults(color=False)
    assert "epochs" in out and "10" in out and "99" in out
    assert "debug" in out
    assert "model.lr" in out
    assert "model.name" in out


def test_format_diff_from_defaults_only_overridden_have_arrows() -> None:
    cfg = Cfg(epochs=99)
    lines = cfg.format_diff_from_defaults(color=False).split("\n")
    epoch_lines = [ln for ln in lines if "epochs" in ln]
    debug_lines = [ln for ln in lines if "debug" in ln]
    assert epoch_lines and "→" in epoch_lines[0]
    assert debug_lines and "→" not in debug_lines[0]


def test_print_diff_from_defaults_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = Cfg(epochs=99)
    cfg.print_diff_from_defaults(color=False)
    captured = capsys.readouterr()
    assert "epochs" in captured.out
    assert "→" in captured.out


def test_format_diff_from_defaults_raises_when_required_fields_missing() -> None:
    cfg = HasRequired(name="explicit")
    with pytest.raises(ValueError, match="required fields"):
        cfg.format_diff_from_defaults()
