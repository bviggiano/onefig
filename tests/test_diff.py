from __future__ import annotations

import pytest

from onefig import MISSING, ConfigModel
from onefig._diff import _MissingType, compute_diff, format_diff


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
    # Keys padded to common width; arrow separator; two-space indent on
    # changed rows leaves room for `+ ` / `- ` markers.
    assert out == (
        "  epochs    10      →  20\n"
        "  model.lr  0.0001  →  0.001"
    )


def test_format_diff_added_row_uses_plus_prefix_no_arrow() -> None:
    out = format_diff({"experiment.id": (MISSING, "abc123")}, color=False)
    assert out == "+ experiment.id  'abc123'"
    assert "→" not in out
    assert "MISSING" not in out


def test_format_diff_removed_row_uses_minus_prefix_no_arrow() -> None:
    out = format_diff({"model.depth": (12, MISSING)}, color=False)
    assert out == "- model.depth  12"
    assert "→" not in out
    assert "MISSING" not in out


def test_format_diff_mixed_rows() -> None:
    diff = {
        "epochs": (10, 20),
        "added": (MISSING, "new"),
        "removed": ("gone", MISSING),
    }
    lines = format_diff(diff, color=False).split("\n")
    assert lines[0].startswith("  ") and "→" in lines[0]
    assert lines[1].startswith("+ ")
    assert lines[2].startswith("- ")


def test_format_diff_changed_row_greens_only_new_value() -> None:
    # Changed rows highlight only the new value in green; the prior
    # value is left uncolored (it's a reference, not a removal).
    out = format_diff({"epochs": (10, 20)}, color=True)
    assert "\033[32m" in out   # green for new
    assert "\033[31m" not in out   # no red on old side
    assert "\033[0m" in out    # reset


def test_format_diff_color_off_strips_ansi() -> None:
    out = format_diff({"epochs": (10, 20)}, color=False)
    assert "\033[" not in out


def test_format_diff_added_row_is_green_when_colored() -> None:
    out = format_diff({"k": (MISSING, "v")}, color=True)
    assert "\033[32m" in out   # green for the added value
    assert "\033[31m" not in out


def test_format_diff_removed_row_is_red_when_colored() -> None:
    # Red is reserved for `-` rows (actual removals).
    out = format_diff({"k": ("v", MISSING)}, color=True)
    assert "\033[31m" in out   # red for the removed value
    assert "\033[32m" not in out


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


def test_format_diff_from_defaults_arrows_point_default_to_current() -> None:
    cfg = Cfg(epochs=99)
    out = cfg.format_diff_from_defaults(color=False)
    # Default 10 → current 99 (arrow reads "was 10, now 99").
    assert "10" in out and "99" in out
    assert out.index("10") < out.index("99")


def test_print_diff_from_defaults_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = Cfg(epochs=99)
    cfg.print_diff_from_defaults(color=False)
    captured = capsys.readouterr()
    assert "epochs" in captured.out
    assert "→" in captured.out
