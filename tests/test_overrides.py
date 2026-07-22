from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import ValidationError, model_validator

from onefig import ConfigError, ConfigModel, tagged_union


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
    with pytest.raises(ConfigError):
        cfg.update_from_args({"lr": "not_a_float"})


def test_overrides_apply_then_validate_once() -> None:
    """All overrides merge before one validation pass, so it's order-independent."""

    class _Rng(ConfigModel):
        lo: float = 0.1
        hi: float = 0.9

        @model_validator(mode="after")
        def _ordered(self) -> _Rng:
            if self.lo >= self.hi:
                raise ValueError("lo must be < hi")
            return self

    # The final state (0.95 < 0.99) is valid; both orders succeed even though an
    # intermediate (lo=0.95 while hi is still 0.9) would violate lo < hi per-field.
    for overrides in ({"lo": 0.95, "hi": 0.99}, {"hi": 0.99, "lo": 0.95}):
        cfg = _Rng()
        cfg.update_from_args(overrides)
        assert (cfg.lo, cfg.hi) == (0.95, 0.99)

    # A genuinely invalid final config still raises.
    with pytest.raises(ConfigError):
        _Rng().update_from_args({"lo": 0.99, "hi": 0.1})


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


def test_attribute_access_requires_full_path() -> None:
    # In code, only full attribute paths resolve; leaf shortcuts are CLI/env-only.
    cfg = _Cfg()
    cfg.model.lr = 0.5
    assert cfg.model.lr == 0.5
    with pytest.raises(AttributeError, match="full path"):
        cfg.lr = 0.5  # leaf write is rejected
    with pytest.raises(AttributeError, match="full path"):
        _ = cfg.lr  # leaf read is rejected


def test_attribute_unknown_name_raises() -> None:
    cfg = _Cfg()
    with pytest.raises(AttributeError):
        _ = cfg.totally_unknown
    with pytest.raises(AttributeError):
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
    with pytest.raises(AttributeError, match="ambiguous"):
        cfg.lr = 0.5
    with pytest.raises(AttributeError, match="ambiguous"):
        _ = cfg.lr


def test_full_path_assignment_revalidates() -> None:
    cfg = _Cfg()
    with pytest.raises(ValidationError):
        cfg.model.lr = "not_a_float"


def test_frozen_blocks_writes() -> None:
    from onefig.model import FrozenConfigError

    cfg = _Cfg()
    cfg.model.lr = 0.42
    cfg.freeze()

    assert cfg.model.lr == 0.42  # reads are unaffected (full path)
    with pytest.raises(FrozenConfigError):
        cfg.epochs = 99
    with pytest.raises(FrozenConfigError):
        cfg.model.lr = 0.5  # freeze() recurses into nested configs


def test_cli_suffix_subpath_resolves() -> None:
    class Inner(ConfigModel):
        dropout: float = 0.0

    class Mid(ConfigModel):
        neck: Inner = Inner()

    class Outer(ConfigModel):
        objective: Mid = Mid()

    cfg = Outer()
    cfg.update_from_args({"neck.dropout": 0.3})  # a suffix subpath, not just the leaf
    assert cfg.objective.neck.dropout == 0.3
    cfg.update_from_args({"dropout": 0.5})  # the bare leaf still works
    assert cfg.objective.neck.dropout == 0.5
    cfg.update_from_args({"objective.neck.dropout": 0.7})  # and the full path
    assert cfg.objective.neck.dropout == 0.7


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


# ---- overrides into dataclass fields ---------------------------------------


@dataclass
class _Adam:
    name: Literal["adam"] = "adam"
    lr: float = 1e-3

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be > 0")


@dataclass
class _SGD:
    name: Literal["sgd"] = "sgd"
    lr: float = 1e-2
    momentum: float = 0.9


_Optimizer = tagged_union(_Adam, _SGD)


class _Trainer(ConfigModel):
    optimizer: _Optimizer = _Adam()
    betas: tuple[float, float] = (0.9, 0.999)


class _HoldsPlainDataclass(ConfigModel):
    # A dataclass that is not part of a union, held directly by a field.
    optimizer: _Adam = _Adam()


def test_override_dataclass_field_via_dotted_path() -> None:
    cfg = _Trainer()
    cfg.update_from_args({"optimizer.lr": 0.5})
    assert isinstance(cfg.optimizer, _Adam)
    assert cfg.optimizer.lr == 0.5


def test_override_dataclass_field_via_leaf_shortcut() -> None:
    cfg = _Trainer()
    cfg.update_from_args({"lr": 0.25})
    assert cfg.optimizer.lr == 0.25


def test_override_dataclass_field_reruns_post_init_bounds() -> None:
    cfg = _Trainer()
    with pytest.raises(ConfigError):
        cfg.update_from_args({"optimizer.lr": -1.0})


def test_override_dataclass_field_coerces_value() -> None:
    cfg = _Trainer()
    cfg.update_from_args({"optimizer.lr": "0.5"})  # string, as CLI would supply
    assert cfg.optimizer.lr == 0.5
    assert isinstance(cfg.optimizer.lr, float)


def test_override_preserves_untouched_dataclass_fields() -> None:
    cfg = _Trainer()
    cfg.optimizer = _SGD()  # select the other variant
    cfg.update_from_args({"momentum": 0.5})
    assert isinstance(cfg.optimizer, _SGD)
    assert cfg.optimizer.momentum == 0.5
    assert cfg.optimizer.lr == 1e-2  # untouched default retained


def test_override_plain_dataclass_field() -> None:
    cfg = _HoldsPlainDataclass()
    cfg.update_from_args({"optimizer.lr": 0.5})
    assert cfg.optimizer.lr == 0.5


def test_dataclass_leaf_offered_as_attribute() -> None:
    cfg = _Trainer()
    cfg.optimizer.lr = 0.02  # direct write on the dataclass instance
    assert cfg.optimizer.lr == 0.02
