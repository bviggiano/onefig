"""Typed, self-validating Weights & Biases sweep configs.

Models the wandb sweep-configuration format
(https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) as onefig
configs, so ``WandbSweepConfig.load("sweep.yaml")`` reports a field-tree error
the moment the file misspecifies the format — a mistyped key, a distribution
missing a required argument, a continuous range under grid search, or a bayes
sweep with no metric. Only the *format* is modeled, so this module never imports
wandb and adds no dependency.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from onefig.model import ConfigModel

# Distributions that draw from a numeric [min, max] range.
_RANGE_DISTS = frozenset(
    {
        "uniform",
        "int_uniform",
        "q_uniform",
        "log_uniform",
        "log_uniform_values",
        "q_log_uniform",
        "q_log_uniform_values",
        "inv_log_uniform",
        "inv_log_uniform_values",
    }
)
# Distributions parameterized by mean/standard deviation.
_NORMAL_DISTS = frozenset({"normal", "q_normal", "log_normal", "q_log_normal"})
# Range distributions in log space, whose bounds must be strictly positive.
_LOG_RANGE_DISTS = frozenset({d for d in _RANGE_DISTS if "log" in d})

Distribution = Literal[
    "constant",
    "categorical",
    "uniform",
    "int_uniform",
    "q_uniform",
    "log_uniform",
    "log_uniform_values",
    "q_log_uniform",
    "q_log_uniform_values",
    "inv_log_uniform",
    "inv_log_uniform_values",
    "normal",
    "q_normal",
    "log_normal",
    "q_log_normal",
]

_SCALAR_KEYS = (
    "value",
    "values",
    "probabilities",
    "distribution",
    "min",
    "max",
    "mu",
    "sigma",
    "q",
)


class SweepParameter(BaseModel):
    """One entry under ``parameters``: a constant, discrete set, range, or nested group.

    Forms are mutually exclusive; the validator reports the conflicting/missing keys.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any = None
    values: list[Any] | None = None
    probabilities: list[float] | None = None
    distribution: Distribution | None = None
    min: float | None = None
    max: float | None = None
    mu: float | None = None
    sigma: float | None = None
    q: float | None = None
    parameters: dict[str, SweepParameter] | None = None

    @property
    def is_discrete(self) -> bool:
        """Whether this parameter enumerates fixed choices (what grid search needs)."""
        if self.parameters is not None:
            return all(child.is_discrete for child in self.parameters.values())
        if self.distribution in _RANGE_DISTS or self.distribution in _NORMAL_DISTS:
            return False
        if self.min is not None or self.max is not None:
            return False
        return (
            self.value is not None
            or self.values is not None
            or self.distribution in ("constant", "categorical")
        )

    def _set_scalar_keys(self) -> set[str]:
        return {key for key in _SCALAR_KEYS if getattr(self, key) is not None}

    def _forbid_beyond(self, allowed: set[str], label: str) -> None:
        extra = self._set_scalar_keys() - allowed
        if extra:
            raise ValueError(f"{label} does not use: {', '.join(sorted(extra))}")

    @model_validator(mode="after")
    def _validate_form(self) -> Self:
        if self.parameters is not None:
            if self._set_scalar_keys():
                raise ValueError("a nested `parameters` group uses no other keys")
            return self

        dist = self.distribution
        if dist == "constant" or (dist is None and self.value is not None):
            self._forbid_beyond({"value", "distribution"}, "a constant parameter")
            if self.value is None:
                raise ValueError("distribution 'constant' requires a `value`")
        elif dist == "categorical" or (dist is None and self.values is not None):
            self._validate_categorical()
        elif dist in _NORMAL_DISTS:
            self._validate_normal(dist)
        elif dist in _RANGE_DISTS or (
            dist is None and (self.min is not None or self.max is not None)
        ):
            self._validate_range(dist)
        else:
            raise ValueError(
                "empty parameter: set `value`, `values`, `min`/`max`, or `distribution`"
            )
        return self

    def _validate_categorical(self) -> None:
        self._forbid_beyond(
            {"values", "probabilities", "distribution"}, "a categorical parameter"
        )
        if not self.values:
            raise ValueError("`values` must be a non-empty list")
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.values):
                raise ValueError("`probabilities` must be the same length as `values`")
            if abs(sum(self.probabilities) - 1.0) > 1e-6:
                raise ValueError("`probabilities` must sum to 1")

    def _validate_range(self, dist: str | None) -> None:
        self._forbid_beyond(
            {"distribution", "min", "max", "q"}, f"distribution {dist or 'uniform'!r}"
        )
        if self.min is None or self.max is None:
            raise ValueError("a numeric range needs both `min` and `max`")
        if self.min >= self.max:
            raise ValueError(f"`min` ({self.min}) must be less than `max` ({self.max})")
        if dist in _LOG_RANGE_DISTS and (self.min <= 0 or self.max <= 0):
            raise ValueError(f"distribution {dist!r} requires `min` and `max` > 0")
        quantized = dist is not None and dist.startswith("q_")
        if quantized and self.q is None:
            raise ValueError(f"distribution {dist!r} requires a quantization step `q`")
        if not quantized and self.q is not None:
            raise ValueError("`q` only applies to a quantized (`q_*`) distribution")

    def _validate_normal(self, dist: str) -> None:
        self._forbid_beyond(
            {"distribution", "mu", "sigma", "q"}, f"distribution {dist!r}"
        )
        if self.mu is None or self.sigma is None:
            raise ValueError(f"distribution {dist!r} requires `mu` and `sigma`")
        if dist.startswith("q_") and self.q is None:
            raise ValueError(f"distribution {dist!r} requires a quantization step `q`")


class SweepMetric(BaseModel):
    """The objective a bayes/early-terminate sweep optimizes."""

    model_config = ConfigDict(extra="forbid")

    name: str
    goal: Literal["minimize", "maximize"] = "minimize"
    target: float | None = None


class EarlyTerminate(BaseModel):
    """Early-stopping policy for a sweep (wandb currently supports Hyperband)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hyperband"]
    min_iter: int | None = None
    max_iter: int | None = None
    s: int | None = None
    eta: float | None = None


class WandbSweepConfig(ConfigModel):
    """A wandb sweep configuration, validated against the wandb sweep format on load.

    ``WandbSweepConfig.load("sweep.yaml")`` raises a onefig field-tree ``ConfigError``
    on any malformed field, before the sweep is ever created. :meth:`to_wandb` returns
    the plain dict wandb's ``sweep()`` API expects.
    """

    method: Literal["grid", "random", "bayes"]
    parameters: dict[str, SweepParameter]
    metric: SweepMetric | None = None
    program: str | None = None
    command: list[str] | None = None
    name: str | None = None
    description: str | None = None
    project: str | None = None
    entity: str | None = None
    early_terminate: EarlyTerminate | None = None
    run_cap: int | None = None

    @model_validator(mode="after")
    def _validate_method(self) -> Self:
        if not self.parameters:
            raise ValueError("`parameters` must define at least one hyperparameter")
        if self.method == "bayes" and self.metric is None:
            raise ValueError("method 'bayes' requires a `metric` to optimize")
        if self.method == "grid":
            for path, param in self.parameters.items():
                if not param.is_discrete:
                    raise ValueError(
                        f"parameter {path!r}: grid search needs `values` (or `value`), "
                        "not a continuous range/distribution"
                    )
        return self

    def to_wandb(self) -> dict[str, Any]:
        """The plain dict ``wandb.sweep()`` expects (unset keys dropped)."""
        return self.model_dump(exclude_none=True)
