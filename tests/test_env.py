from __future__ import annotations

import pytest

from onefig import ConfigModel
from onefig._env import parse_env


class Sub(ConfigModel):
    lr: float = 1e-4
    name: str = "default"


class Cfg(ConfigModel):
    epochs: int = 10
    debug: bool = False
    model: Sub = Sub()


# ---- low-level parse_env ---------------------------------------------------


def test_parse_env_strips_prefix_and_lowercases() -> None:
    out = parse_env({"MYAPP_EPOCHS": "20"}, prefix="MYAPP_")
    assert out == {"epochs": 20}


def test_parse_env_double_underscore_becomes_dot() -> None:
    out = parse_env({"MYAPP_MODEL__LR": "0.001"}, prefix="MYAPP_")
    assert out == {"model.lr": 0.001}


def test_parse_env_ignores_non_matching() -> None:
    out = parse_env(
        {"MYAPP_EPOCHS": "20", "PATH": "/usr/bin", "OTHER_LR": "1"},
        prefix="MYAPP_",
    )
    assert out == {"epochs": 20}


def test_parse_env_case_sensitive_preserves_case() -> None:
    out = parse_env(
        {"MYAPP_Model__LR": "0.001"},
        prefix="MYAPP_",
        case_sensitive=True,
    )
    assert out == {"Model.LR": 0.001}


def test_parse_env_custom_delimiter() -> None:
    out = parse_env(
        {"MYAPP_MODEL___LR": "0.001"},  # ___ as delimiter
        prefix="MYAPP_",
        delimiter="___",
    )
    assert out == {"model.lr": 0.001}


def test_parse_env_coerces_json_values() -> None:
    out = parse_env(
        {
            "X_INT": "5",
            "X_FLOAT": "5.0",
            "X_BOOL": "true",
            "X_NULL": "null",
            "X_LIST": "[1,2,3]",
            "X_STR": "hello",
        },
        prefix="X_",
    )
    assert out == {
        "int": 5,
        "float": 5.0,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "str": "hello",
    }


def test_parse_env_skips_bare_prefix() -> None:
    # An env var whose name equals the prefix exactly has nothing to address.
    out = parse_env({"MYAPP_": "junk", "MYAPP_LR": "0.1"}, prefix="MYAPP_")
    assert out == {"lr": 0.1}


def test_parse_env_rejects_empty_segment() -> None:
    # Trailing delimiter → trailing empty segment.
    with pytest.raises(ValueError, match="empty key segment"):
        parse_env({"MYAPP_MODEL__": "x"}, prefix="MYAPP_")


def test_parse_env_empty_prefix_reads_everything() -> None:
    out = parse_env({"FOO": "1", "BAR": "2"}, prefix="")
    assert out == {"foo": 1, "bar": 2}


# ---- ConfigModel.update_from_env -------------------------------------------


def test_update_from_env_basic() -> None:
    cfg = Cfg()
    cfg.update_from_env("MYAPP_", environ={"MYAPP_EPOCHS": "20"})
    assert cfg.epochs == 20


def test_update_from_env_nested() -> None:
    cfg = Cfg()
    cfg.update_from_env("MYAPP_", environ={"MYAPP_MODEL__LR": "0.001"})
    assert cfg.model.lr == 0.001


def test_update_from_env_leaf_shortcut() -> None:
    # `lr` only exists at `model.lr`, so the leaf shortcut resolves.
    cfg = Cfg()
    cfg.update_from_env("MYAPP_", environ={"MYAPP_LR": "0.5"})
    assert cfg.model.lr == 0.5


def test_update_from_env_strict_rejects_unknown() -> None:
    cfg = Cfg()
    with pytest.raises(ValueError, match="not found"):
        cfg.update_from_env("MYAPP_", environ={"MYAPP_NOPE": "1"})


def test_update_from_env_non_strict_skips_unknown() -> None:
    cfg = Cfg()
    cfg.update_from_env(
        "MYAPP_",
        environ={"MYAPP_NOPE": "1", "MYAPP_EPOCHS": "7"},
        strict=False,
    )
    assert cfg.epochs == 7


def test_update_from_env_type_validation() -> None:
    from onefig import ConfigError

    cfg = Cfg()
    with pytest.raises(ConfigError):
        cfg.update_from_env("MYAPP_", environ={"MYAPP_EPOCHS": "not_an_int"})


def test_update_from_env_bool_coercion() -> None:
    cfg = Cfg()
    cfg.update_from_env("MYAPP_", environ={"MYAPP_DEBUG": "true"})
    assert cfg.debug is True


def test_update_from_env_defaults_to_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYAPP_EPOCHS", "42")
    cfg = Cfg()
    cfg.update_from_env("MYAPP_")
    assert cfg.epochs == 42


def test_update_from_env_composes_with_cli() -> None:
    # The classic precedence chain: YAML defaults → env → CLI.
    cfg = Cfg(epochs=1)
    cfg.update_from_env("MYAPP_", environ={"MYAPP_EPOCHS": "10"})
    assert cfg.epochs == 10
    cfg.update_from_cli(["epochs=20"])
    assert cfg.epochs == 20
