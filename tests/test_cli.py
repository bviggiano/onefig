from __future__ import annotations

import pytest

from onefig import ConfigModel
from onefig._cli import parse_overrides


class _Sub(ConfigModel):
    lr: float = 1e-4
    name: str = "default"


class _Cfg(ConfigModel):
    epochs: int = 10
    model: _Sub = _Sub()
    tags: list[str] = []


def test_parse_overrides_simple() -> None:
    assert parse_overrides(["epochs=5", "lr=0.001"]) == {"epochs": 5, "lr": 0.001}


def test_parse_overrides_string_fallback() -> None:
    assert parse_overrides(["model.name=bert"]) == {"model.name": "bert"}


def test_parse_overrides_bool_and_none() -> None:
    out = parse_overrides(["a=true", "b=false", "c=null", "d=None"])
    assert out == {"a": True, "b": False, "c": None, "d": None}


def test_parse_overrides_list_value() -> None:
    assert parse_overrides(["tags=[1,2,3]"]) == {"tags": [1, 2, 3]}


def test_parse_overrides_missing_equals_raises() -> None:
    with pytest.raises(ValueError, match="expected key=value"):
        parse_overrides(["foo"])


def test_parse_overrides_empty_key_raises() -> None:
    with pytest.raises(ValueError, match="empty key"):
        parse_overrides(["=5"])


def test_update_from_cli_basic() -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["epochs=5", "model.lr=0.001"])
    assert cfg.epochs == 5
    assert cfg.model.lr == 0.001


def test_update_from_cli_leaf_key() -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["lr=0.5"])
    assert cfg.model.lr == 0.5


def test_update_from_cli_strict_unknown_raises() -> None:
    cfg = _Cfg()
    with pytest.raises(ValueError, match="not found"):
        cfg.update_from_cli(["nope=1"])


def test_update_from_cli_non_strict_skips_unknown() -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["nope=1", "epochs=3"], strict=False)
    assert cfg.epochs == 3


def test_update_from_cli_uses_argv_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["script.py", "epochs=7"])
    cfg = _Cfg()
    cfg.update_from_cli()
    assert cfg.epochs == 7


def test_update_from_cli_revalidates() -> None:
    from onefig import ConfigError

    cfg = _Cfg()
    with pytest.raises(ConfigError):
        cfg.update_from_cli(["epochs=not_an_int"])


def test_update_from_cli_show_prints_and_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    with pytest.raises(SystemExit) as exc_info:
        cfg.update_from_cli(["epochs=3", "--show"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "epochs" in out
    # Overrides applied before display, so the printed value should match.
    assert "3" in out


def test_update_from_cli_show_without_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["epochs=3", "--show"], exit_on_show=False)
    assert cfg.epochs == 3
    assert "epochs" in capsys.readouterr().out


def test_update_from_cli_no_show_does_not_print(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["epochs=3"])
    assert capsys.readouterr().out == ""
