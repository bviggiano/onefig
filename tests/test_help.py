from __future__ import annotations

import enum
from typing import Literal

import pytest
from pydantic import Field

from onefig import ConfigModel
from onefig._help import format_help


class _Optimizer(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "sgd"
    """Which optimizer to use."""
    lr: float = 1e-4
    """Learning rate."""


class _Loss(str, enum.Enum):
    CE = "cross_entropy"
    MSE = "mse"


class _Cfg(ConfigModel):
    epochs: int = 10
    """Number of training epochs."""
    optimizer: _Optimizer = _Optimizer()
    loss: _Loss = _Loss.CE
    """Loss function."""
    tags: list[str] = []
    name: str | None = None
    explicit: int = Field(default=7, description="Explicit field description.")
    """This docstring should be ignored in favor of the explicit description."""


def test_help_includes_field_paths() -> None:
    out = format_help(_Cfg())
    assert "epochs" in out
    # Nested fields appear under their panel title with the leaf name only;
    # the full dotted path is implied by the panel header + entry combo.
    assert "╭─ optimizer " in out
    assert "kind : " in out
    assert "lr : float" in out


def test_help_shows_default_and_current() -> None:
    cfg = _Cfg()
    cfg.epochs = 99
    out = format_help(cfg)
    # default and current both shown for epochs
    assert "default: 10" in out
    assert "current: 99" in out


def test_help_renders_literal_choices() -> None:
    out = format_help(_Cfg())
    assert "{'sgd', 'adam', 'adamw'}" in out


def test_help_renders_enum_choices() -> None:
    out = format_help(_Cfg())
    assert "_Loss" in out
    assert "'cross_entropy'" in out
    assert "'mse'" in out


def test_help_includes_attribute_docstrings() -> None:
    out = format_help(_Cfg())
    assert "Number of training epochs." in out
    assert "Which optimizer to use." in out


def test_description_starts_on_a_new_indented_line() -> None:
    """Description always begins on its own hang-indented line below the head."""

    class Tiny(ConfigModel):
        epochs: int = 10
        """Number of epochs."""

    out = format_help(Tiny())
    lines = out.splitlines()
    head_idx = next(
        i
        for i, line in enumerate(lines)
        if "epochs : int" in line and "(default: 10" in line
    )
    next_line = lines[head_idx + 1]
    inner = next_line.strip("│").rstrip()
    leading_spaces = len(inner) - len(inner.lstrip(" "))
    assert leading_spaces >= 6, next_line
    assert "Number of epochs." in next_line
    # The pipe separator is no longer used in the layout.
    assert " | " not in lines[head_idx]


def test_long_description_overflow_stays_tab_indented() -> None:
    """Each continuation line of a wrapped description keeps the tab indent."""

    class Verbose(ConfigModel):
        x: int = 1
        """A very long description that goes on and on with many words deliberately
        to force textwrap into producing more than one continuation line so we can
        verify the tab indent behavior on the second and third lines too."""

    out = format_help(Verbose())
    lines = out.splitlines()
    head_idx = next(i for i, line in enumerate(lines) if "x : int" in line)
    desc_lines = []
    for line in lines[head_idx + 1 :]:
        inner = line.strip("│").rstrip()
        if not inner:
            break
        desc_lines.append(line)
        leading = len(inner) - len(inner.lstrip(" "))
        assert leading >= 6, line
    # Verbose text should produce at least two wrapped description lines.
    assert len(desc_lines) >= 2, desc_lines


def test_field_description_takes_precedence_over_docstring() -> None:
    out = format_help(_Cfg())
    assert "Explicit field description." in out
    assert "This docstring should be ignored" not in out


def test_help_renders_optional_type() -> None:
    out = format_help(_Cfg())
    # Optional[str] -> "str | None"
    assert "str | None" in out


def test_help_renders_list_type() -> None:
    out = format_help(_Cfg())
    assert "list[str]" in out


def test_help_title_defaults_to_class_name() -> None:
    out = format_help(_Cfg())
    first_line = out.splitlines()[0]
    assert "_Cfg" in first_line
    assert first_line.startswith("╭")


def test_help_title_override() -> None:
    out = format_help(_Cfg(), title="MyExperiment")
    assert "MyExperiment" in out.splitlines()[0]


def test_help_renders_panels_with_rounded_corners() -> None:
    out = format_help(_Cfg()).splitlines()
    assert out[0].startswith("╭") and out[0].endswith("╮")
    assert out[-1].startswith("╰") and out[-1].endswith("╯")
    # Non-empty body lines are bordered with │ or ├ on both ends; empty
    # lines are the blank separators between panels.
    body = out[1:-1]
    for line in body:
        if line == "":
            continue
        assert line.startswith(("│", "├", "╭", "╰")), line
        assert line.endswith(("│", "┤", "╮", "╯")), line


def test_nested_configs_render_as_their_own_panels() -> None:
    out = format_help(_Cfg())
    # The top panel and the nested optimizer panel each get their own banner.
    assert out.count("╭─ _Cfg ") == 1
    assert out.count("╭─ optimizer ") == 1
    assert out.count("╭─ flags ") == 1


def test_unique_leaf_shows_just_the_leaf_name() -> None:
    out = format_help(_Cfg())
    # 'kind' is unique to optimizer.kind, so the entry shows only "kind".
    # The panel title 'optimizer' already provides the parent context.
    assert "kind : {'sgd', 'adam', 'adamw'}" in out
    assert "lr : float" in out
    # No annotated full-path suffix anywhere.
    assert "(optimizer.kind)" not in out
    assert "(optimizer.lr)" not in out


def test_top_level_field_does_not_repeat_path() -> None:
    out = format_help(_Cfg())
    assert "epochs : int" in out
    assert "epochs (epochs)" not in out


def test_ambiguous_leaf_shows_full_path() -> None:
    class A(ConfigModel):
        lr: float = 1e-4

    class B(ConfigModel):
        lr: float = 1e-3

    class TwoOpt(ConfigModel):
        a: A = A()
        b: B = B()

    out = format_help(TwoOpt())
    # 'lr' is ambiguous (matches a.lr and b.lr), so each entry uses its
    # full dotted path instead of the bare leaf.
    assert "a.lr : float" in out
    assert "b.lr : float" in out
    assert "(a.lr)" not in out
    assert "(b.lr)" not in out


def test_help_lists_special_flags() -> None:
    out = format_help(_Cfg())
    assert "--show" in out
    assert "--help" in out
    assert "-h" in out


def test_required_field_has_no_default_line() -> None:
    class Req(ConfigModel):
        x: int  # required, no default

    out = format_help(Req(x=5))
    # Required fields shouldn't show "default:" anywhere.
    assert "default:" not in out
    assert "current: 5" in out
    assert "x : int" in out


def test_print_help_method(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _Cfg()
    cfg.print_help()
    out = capsys.readouterr().out
    assert "epochs" in out
    assert "Number of training epochs." in out


def test_format_help_method_uses_config_name() -> None:
    cfg = _Cfg()
    cfg.config_name = "my_run"
    out = cfg.format_help()
    assert "my_run" in out.splitlines()[0]


def test_format_help_explicit_title_overrides_config_name() -> None:
    cfg = _Cfg()
    cfg.config_name = "my_run"
    out = cfg.format_help(title="Override")
    assert "Override" in out.splitlines()[0]
    assert "my_run" not in out.splitlines()[0]


def test_help_flag_in_update_from_cli_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    with pytest.raises(SystemExit) as exc_info:
        cfg.update_from_cli(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "epochs" in out
    assert "╭─ optimizer " in out
    assert "kind : " in out


def test_short_help_flag_in_update_from_cli_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    with pytest.raises(SystemExit) as exc_info:
        cfg.update_from_cli(["-h"])
    assert exc_info.value.code == 0
    assert "epochs" in capsys.readouterr().out


def test_help_flag_short_circuits_overrides() -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["epochs=99", "--help"], exit_on_help=False)
    # --help short-circuits: epochs should NOT have been updated
    assert cfg.epochs == 10


def test_help_flag_without_exit(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["--help"], exit_on_help=False)
    assert "epochs" in capsys.readouterr().out


def test_help_flag_does_not_consume_unknown_overrides_when_help_passed() -> None:
    """--help short-circuits before strict-mode override validation."""
    cfg = _Cfg()
    # 'nope' would normally raise; --help should bypass that.
    cfg.update_from_cli(["nope=1", "--help"], exit_on_help=False)
