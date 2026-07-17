"""WandbSweepConfig validates the wandb sweep format on construction/load."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from onefig.integrations.wandb import WandbSweepConfig

VALID = [
    # bayes with a metric and a mix of distribution / categorical / constant parameters
    {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "optimizer.lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            "optimizer.name": {"values": ["adamw", "adam"]},
            "trainer.batch_size": {"value": 16},
        },
    },
    # grid over discrete choices only
    {"method": "grid", "parameters": {"trainer.batch_size": {"values": [8, 16, 32]}}},
    # random with implicit int range and an explicit uniform
    {
        "method": "random",
        "parameters": {
            "trainer.epochs": {"min": 1, "max": 5},
            "optimizer.weight_decay": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.1,
            },
        },
    },
    # normal distribution and a quantized range
    {
        "method": "random",
        "parameters": {
            "a": {"distribution": "normal", "mu": 0.0, "sigma": 1.0},
            "b": {"distribution": "q_uniform", "min": 0, "max": 10, "q": 2},
        },
    },
    # categorical with probabilities, plus early termination
    {
        "method": "bayes",
        "metric": {"name": "val/accuracy", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {"opt": {"values": ["a", "b"], "probabilities": [0.7, 0.3]}},
    },
    # nested parameter group
    {
        "method": "grid",
        "parameters": {"group": {"parameters": {"x": {"values": [1, 2]}}}},
    },
]

INVALID = [
    ({"method": "bayes", "parameters": {"x": {"value": 1}}}, "requires a `metric`"),
    (
        {
            "method": "grid",
            "parameters": {"lr": {"distribution": "uniform", "min": 0, "max": 1}},
        },
        "grid search needs",
    ),
    (
        {
            "method": "random",
            "parameters": {
                "lr": {"distribution": "log_uniform_values", "min": 0, "max": 1}
            },
        },
        "> 0",
    ),
    (
        {"method": "random", "parameters": {"lr": {"min": 1, "max": 10, "mu": 0.5}}},
        "does not use",
    ),
    (
        {
            "method": "random",
            "parameters": {"lr": {"distribution": "normal", "min": 1, "max": 2}},
        },
        "does not use",
    ),
    (
        {"method": "random", "parameters": {"lr": {"distribution": "normal", "mu": 0}}},
        "requires `mu` and `sigma`",
    ),
    (
        {
            "method": "random",
            "parameters": {
                "lr": {"distribution": "q_log_uniform_values", "min": 1, "max": 9}
            },
        },
        "quantization step",
    ),
    ({"method": "grid", "parameters": {"lr": {"values": []}}}, "non-empty"),
    (
        {
            "method": "grid",
            "parameters": {"o": {"values": ["a", "b"], "probabilities": [1.0]}},
        },
        "same length",
    ),
    (
        {
            "method": "grid",
            "parameters": {"o": {"values": ["a", "b"], "probabilities": [0.5, 0.9]}},
        },
        "sum to 1",
    ),
    ({"method": "random", "parameters": {"lr": {"min": 10, "max": 1}}}, "less than"),
    (
        {"method": "random", "parameters": {"lr": {"minn": 1, "max": 10}}},
        "Extra inputs",
    ),
    ({"method": "random", "parameters": {"lr": {}}}, "empty parameter"),
    (
        {
            "method": "random",
            "parameters": {"lr": {"distribution": "loguniform", "min": 1, "max": 2}},
        },
        "distribution",
    ),
    ({"method": "sobol", "parameters": {"lr": {"values": [1]}}}, "method"),
    ({"method": "random", "parameters": {}}, "at least one"),
    (
        {
            "method": "random",
            "parameters": {
                "lr": {"distribution": "uniform", "min": 0, "max": 1, "q": 2}
            },
        },
        "quantized",
    ),
]


@pytest.mark.parametrize("cfg", VALID)
def test_valid_sweeps_construct(cfg: dict) -> None:
    WandbSweepConfig.model_validate(cfg)


@pytest.mark.parametrize("cfg,message", INVALID)
def test_invalid_sweeps_are_rejected(cfg: dict, message: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        WandbSweepConfig.model_validate(cfg)
    assert message in str(excinfo.value)


def test_to_wandb_drops_unset_keys() -> None:
    cfg = WandbSweepConfig.model_validate(
        {"method": "grid", "parameters": {"x": {"values": [1, 2]}}}
    )
    out = cfg.to_wandb()
    assert out == {"method": "grid", "parameters": {"x": {"values": [1, 2]}}}
    assert "metric" not in out and "program" not in out  # None fields are omitted


def test_load_reports_field_tree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from onefig import ConfigError

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "method: bayes\nparameters:\n  lr:\n    value: 1\n"
    )  # bayes without a metric
    with pytest.raises(ConfigError) as excinfo:
        WandbSweepConfig.load(str(sweep))
    assert "metric" in str(excinfo.value)
