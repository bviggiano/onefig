from __future__ import annotations

import pytest

from onefig import ConfigModel, flatten, unflatten


def test_flatten_simple() -> None:
    assert flatten({"a": 1, "b": {"c": 2}}) == {"a": 1, "b.c": 2}


def test_flatten_list_of_scalars() -> None:
    assert flatten({"tags": ["x", "y"]}) == {"tags.0": "x", "tags.1": "y"}


def test_flatten_list_of_dicts() -> None:
    data = {"layers": [{"size": 768}, {"size": 1024}]}
    assert flatten(data) == {"layers.0.size": 768, "layers.1.size": 1024}


def test_unflatten_simple() -> None:
    assert unflatten({"a": 1, "b.c": 2}) == {"a": 1, "b": {"c": 2}}


def test_unflatten_list_of_scalars() -> None:
    assert unflatten({"tags.0": "x", "tags.1": "y"}) == {"tags": ["x", "y"]}


def test_unflatten_list_of_dicts() -> None:
    flat = {"layers.0.size": 768, "layers.1.size": 1024}
    assert unflatten(flat) == {"layers": [{"size": 768}, {"size": 1024}]}


def test_unflatten_handles_out_of_order_indices() -> None:
    flat = {"layers.1.size": 1024, "layers.0.size": 768}
    assert unflatten(flat) == {"layers": [{"size": 768}, {"size": 1024}]}


def test_unflatten_sparse_indices_raises() -> None:
    with pytest.raises(ValueError, match="sparse"):
        unflatten({"tags.0": "x", "tags.2": "z"})


def test_unflatten_conflict_leaf_and_parent_raises() -> None:
    with pytest.raises(ValueError, match="Conflict"):
        unflatten({"a": 1, "a.b": 2})


def test_unflatten_duplicate_path_raises() -> None:
    # `"a.b"` makes node["a"] a dict; then `"a"` tries to overwrite it.
    with pytest.raises(ValueError, match="Duplicate"):
        unflatten({"a.b": 1, "a": 2})


def test_roundtrip_dict_only() -> None:
    data = {
        "epochs": 10,
        "model": {"name": "bert", "hidden_size": 768},
        "optimizer": {"name": "adamw", "lr": 1e-4},
    }
    assert unflatten(flatten(data)) == data


def test_roundtrip_with_list_of_scalars() -> None:
    data = {"epochs": 3, "tags": ["alpha", "beta", "gamma"]}
    assert unflatten(flatten(data)) == data


def test_roundtrip_with_list_of_dicts() -> None:
    data = {
        "name": "x",
        "layers": [{"size": 768, "kind": "attn"}, {"size": 1024, "kind": "ffn"}],
    }
    assert unflatten(flatten(data)) == data


def test_roundtrip_deeply_nested() -> None:
    data = {
        "a": {"b": {"c": {"d": 1, "e": [{"f": 2}, {"f": 3}]}}},
        "g": [10, 20, 30],
    }
    assert unflatten(flatten(data)) == data


# ---------------------------------------------------------------------------
# Round-trip via ConfigModel
# ---------------------------------------------------------------------------


class _Inner(ConfigModel):
    name: str = "bert"
    lr: float = 1e-4


class _Outer(ConfigModel):
    epochs: int = 10
    inner: _Inner = _Inner()
    tags: list[str] = []


def test_config_roundtrip_through_flat_dict() -> None:
    cfg = _Outer(epochs=5, inner=_Inner(name="custom", lr=0.001), tags=["a", "b"])
    flat = cfg.to_flat_dict()
    rebuilt = _Outer.from_flat_dict(flat)
    assert rebuilt == cfg


def test_config_from_flat_dict_validates() -> None:
    from onefig import ConfigError

    with pytest.raises(ConfigError):
        _Outer.from_flat_dict({"epochs": "not_an_int"})


def test_config_from_flat_dict_extra_forbidden() -> None:
    from onefig import ConfigError

    with pytest.raises(ConfigError):
        _Outer.from_flat_dict({"epochs": 1, "unknown.field": 2})
