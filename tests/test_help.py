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
    assert "optimizer.kind" in out
    assert "optimizer.lr" in out


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
    assert out.startswith("_Cfg")


def test_help_title_override() -> None:
    out = format_help(_Cfg(), title="MyExperiment")
    assert out.startswith("MyExperiment")


def test_help_lists_special_flags() -> None:
    out = format_help(_Cfg())
    assert "--show" in out
    assert "--help" in out
    assert "-h" in out


def test_required_field_has_no_default_line() -> None:
    class Req(ConfigModel):
        x: int  # required, no default

    out = format_help(Req(x=5))
    # Required fields shouldn't show "default:"
    field_block = [line for line in out.splitlines() if line.strip().startswith("x ")]
    assert field_block, out
    assert "default:" not in field_block[0]
    assert "current: 5" in field_block[0]


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
    assert out.startswith("my_run")


def test_format_help_explicit_title_overrides_config_name() -> None:
    cfg = _Cfg()
    cfg.config_name = "my_run"
    out = cfg.format_help(title="Override")
    assert out.startswith("Override")


def test_help_flag_in_update_from_cli_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    with pytest.raises(SystemExit) as exc_info:
        cfg.update_from_cli(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "epochs" in out
    assert "optimizer.kind" in out


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
