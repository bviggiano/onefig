from __future__ import annotations

import argparse
from argparse import Namespace

import pytest

from onefig import ConfigModel


class _Sub(ConfigModel):
    lr: float = 1e-4
    name: str = "x"


class _Cfg(ConfigModel):
    epochs: int = 10
    model: _Sub = _Sub()


def test_leaf_key_override() -> None:
    cfg = _Cfg()
    cfg.update_from_args(Namespace(lr=0.5, epochs=100))
    assert cfg.model.lr == 0.5
    assert cfg.epochs == 100


def test_dotted_path_override() -> None:
    cfg = _Cfg()
    cfg.update_from_args({"model.lr": 0.7})
    assert cfg.model.lr == 0.7


def test_skip_none_default() -> None:
    cfg = _Cfg()
    cfg.update_from_args(Namespace(lr=None, epochs=5))
    assert cfg.model.lr == 1e-4
    assert cfg.epochs == 5


def test_unknown_key_strict_raises() -> None:
    cfg = _Cfg()
    with pytest.raises(ValueError, match="not found"):
        cfg.update_from_args({"nope": 1})


def test_unknown_key_non_strict_skips() -> None:
    cfg = _Cfg()
    cfg.update_from_args({"nope": 1, "epochs": 3}, strict=False)
    assert cfg.epochs == 3


def test_ambiguous_leaf_key_raises() -> None:
    class A(ConfigModel):
        lr: float = 1e-4

    class B(ConfigModel):
        lr: float = 1e-3

    class C(ConfigModel):
        a: A = A()
        b: B = B()

    cfg = C()
    with pytest.raises(ValueError, match="Ambiguous"):
        cfg.update_from_args({"lr": 0.5})


def test_dotted_resolves_ambiguity() -> None:
    class A(ConfigModel):
        lr: float = 1e-4

    class B(ConfigModel):
        lr: float = 1e-3

    class C(ConfigModel):
        a: A = A()
        b: B = B()

    cfg = C()
    cfg.update_from_args({"a.lr": 0.9})
    assert cfg.a.lr == 0.9
    assert cfg.b.lr == 1e-3


def test_override_revalidates() -> None:
    cfg = _Cfg()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cfg.update_from_args({"lr": "not_a_float"})


def test_argparse_leaf_keys() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(["--lr", "0.25"])

    cfg = _Cfg()
    cfg.update_from_args(args)
    assert cfg.model.lr == 0.25
    assert cfg.epochs == 10  # epochs was None; skip_none kept the default


def test_argparse_dotted_path_via_dest() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lr", dest="model.lr", type=float)
    args = parser.parse_args(["--model-lr", "0.99"])

    cfg = _Cfg()
    cfg.update_from_args(args)
    assert cfg.model.lr == 0.99


def test_attribute_write_uses_leaf_key_shortcut() -> None:
    cfg = _Cfg()
    cfg.lr = 0.5
    assert cfg.model.lr == 0.5


def test_attribute_read_uses_leaf_key_shortcut() -> None:
    cfg = _Cfg()
    cfg.model.lr = 0.25
    assert cfg.lr == 0.25


def test_attribute_unknown_name_still_raises() -> None:
    from pydantic import ValidationError

    cfg = _Cfg()
    with pytest.raises(AttributeError):
        _ = cfg.totally_unknown
    with pytest.raises(ValidationError):
        cfg.totally_unknown = 1


def test_attribute_ambiguous_leaf_raises() -> None:
    class A(ConfigModel):
        lr: float = 1e-4

    class B(ConfigModel):
        lr: float = 1e-3

    class C(ConfigModel):
        a: A = A()
        b: B = B()

    cfg = C()
    with pytest.raises(AttributeError, match="Ambiguous"):
        cfg.lr = 0.5
    with pytest.raises(AttributeError, match="Ambiguous"):
        _ = cfg.lr


def test_attribute_shortcut_revalidates() -> None:
    from pydantic import ValidationError

    cfg = _Cfg()
    with pytest.raises(ValidationError):
        cfg.lr = "not_a_float"


def test_frozen_blocks_shortcut_writes_but_allows_reads() -> None:
    from onefig.model import FrozenConfigError

    cfg = _Cfg()
    cfg.model.lr = 0.42
    cfg.freeze()

    # Reads still work — freezing doesn't change data.
    assert cfg.lr == 0.42

    # Writes (direct, dotted, and leaf-shortcut) all raise.
    with pytest.raises(FrozenConfigError):
        cfg.lr = 0.5
    with pytest.raises(FrozenConfigError):
        cfg.epochs = 99
    with pytest.raises(FrozenConfigError):
        cfg.model.lr = 0.5


def test_top_level_full_path_beats_nested_leaf() -> None:
    class Inner(ConfigModel):
        lr: float = 0.1

    class Outer(ConfigModel):
        lr: float = 0.5
        inner: Inner = Inner()

    cfg = Outer()
    cfg.update_from_args({"lr": 0.9})  # matches top-level "lr" full path
    assert cfg.lr == 0.9
    assert cfg.inner.lr == 0.1
