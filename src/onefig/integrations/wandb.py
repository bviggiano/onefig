"""Typed, self-validating Weights & Biases sweep configs.

Models the wandb sweep-configuration format
(https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) as onefig
configs, so ``WandbSweepConfig.load("sweep.yaml")`` reports a field-tree error
the moment the file misspecifies the format — a mistyped key, a distribution
missing a required argument, a continuous range under grid search, or a bayes
sweep with no metric. Only the *format* is modeled, so this module never imports
wandb and adds no dependency.

Grid and random sweeps are fully decided before any trial runs, so
:meth:`WandbSweepConfig.trials` lists their trials without wandb's sweep server.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import random
from collections.abc import Iterable, Iterator
from itertools import product
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import Self

from onefig._errors import ConfigError
from onefig._overrides import apply_overrides, resolve_keys
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
# Range distributions WandbSweepConfig.trials draws from.
_DRAWN_DISTS = ("uniform", "int_uniform", "log_uniform_values")

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
    # Kept as written: wandb draws a range whose bounds are both ints as ints.
    min: int | float | None = None
    max: int | float | None = None
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

    def candidate_values(self) -> list[tuple[Any, str]]:
        """Representative ``(value, role)`` pairs to probe against a target field.

        The endpoints/choices where a target's bounds or Literal set would be
        violated: each of ``values``, the constant ``value``, the ``min``/``max``
        of a range, or the ``mu`` of a normal distribution.
        """
        if self.value is not None:
            return [(self.value, "value")]
        if self.values is not None:
            return [(item, "value") for item in self.values]
        out: list[tuple[Any, str]] = []
        if self.min is not None:
            out.append((self.min, "min"))
        if self.max is not None:
            out.append((self.max, "max"))
        if self.mu is not None:
            out.append((self.mu, "mu"))
        return out

    def sample(self, rng: random.Random) -> Any:
        """Draw one value as wandb's random search would.

        A range without a ``distribution`` is ``int_uniform`` when both bounds are
        ints and ``uniform`` otherwise, as in wandb.

        Raises:
            ConfigError: For a nested group, or a distribution other than
                ``constant``, ``categorical``, ``uniform``, ``int_uniform`` or
                ``log_uniform_values``.
        """
        if self.parameters is not None:
            raise ConfigError("a nested `parameters` group draws each child instead")
        if self.value is not None:
            return self.value
        if self.values is not None:
            return rng.choices(self.values, weights=self.probabilities)[0]
        low, high = self.min, self.max
        if low is None or high is None:
            raise ConfigError(f"distribution {self.distribution!r} is not drawn here")
        integers = isinstance(low, int) and isinstance(high, int)
        dist = self.distribution or ("int_uniform" if integers else "uniform")
        if dist == "int_uniform":
            return rng.randint(math.ceil(low), math.floor(high))
        if dist == "uniform":
            return rng.uniform(low, high)
        if dist == "log_uniform_values":
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        raise ConfigError(
            f"distribution {dist!r} is not drawn here, only {', '.join(_DRAWN_DISTS)}"
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

    def trials(
        self, count: int | None = None, *, seed: int | str | None = None
    ) -> list[dict[str, Any]]:
        """Every trial a grid or random sweep runs, as ``{dotted path: value}``.

        Grid search runs every combination of each parameter's ``values`` (or
        ``value``), in parameter-path order. Random search draws ``count`` trials,
        ``run_cap`` by default, each parameter as :meth:`SweepParameter.sample`
        draws it. The draws are seeded by the sweep itself unless ``seed`` is
        given, so the same sweep always lists the same trials. A nested group's
        parameters are keyed by their dotted path, as in :meth:`validate_against`.

        Raises:
            ConfigError: For a sweep that needs wandb's sweep server (bayes search,
                which picks each trial from the results before it, or
                ``early_terminate``, which stops trials by comparing them as they
                run), a random sweep with neither ``count`` nor ``run_cap``, or a
                distribution :meth:`SweepParameter.sample` does not draw.
        """
        if self.method == "bayes":
            raise ConfigError(
                "bayes search picks each trial from the results of the ones before it,"
                " so its trials can't be listed in advance"
            )
        if self.early_terminate is not None:
            raise ConfigError(
                "early_terminate stops trials by comparing their results as they run,"
                " so a sweep that uses it can't be listed in advance"
            )
        leaves = sorted(self._leaf_parameters(), key=lambda leaf: leaf[0])
        paths = [path for path, _ in leaves]
        if self.method == "grid":
            choices = [
                [param.value] if param.values is None else param.values
                for _, param in leaves
            ]
            return [dict(zip(paths, combination)) for combination in product(*choices)]
        count = self.run_cap if count is None else count
        if count is None:
            raise ConfigError(
                "a random sweep's trials need a count: pass one or set `run_cap`"
            )
        if seed is None:
            seed = json.dumps(self.to_wandb(), sort_keys=True, default=str)
        rng = random.Random(hashlib.sha256(str(seed).encode()).digest())
        undrawn = [
            f"{path} ({param.distribution})"
            for path, param in leaves
            if param.distribution
            not in (None, "constant", "categorical", *_DRAWN_DISTS)
        ]
        if undrawn:
            raise ConfigError(
                f"only {', '.join(_DRAWN_DISTS)} ranges are drawn here, not "
                + ", ".join(undrawn)
            )
        return [
            {path: param.sample(rng) for path, param in leaves} for _ in range(count)
        ]

    def _leaf_parameters(self) -> Iterator[tuple[str, SweepParameter]]:
        return _leaves(self.parameters, "")

    def validate_against(
        self, base: ConfigModel, *, logged_metrics: Iterable[str] | None = None
    ) -> None:
        """Check this sweep's parameters against a concrete config instance ``base``.

        Every swept value is applied to a copy of ``base`` and run through its
        validation, so a value outside a field's bounds, an invalid Literal choice,
        a wrong type, or a broken cross-field constraint is caught — along with
        parameter *paths* that don't exist on ``base`` (with a suggestion). When
        ``logged_metrics`` is given, ``metric.name`` is checked against it too.
        Raises :class:`ConfigError` listing every problem found; a no-op if the
        sweep is fully compatible.
        """
        valid_paths = {path for paths in resolve_keys(base).values() for path in paths}
        problems: list[str] = []
        for path, param in self._leaf_parameters():
            if path not in valid_paths:
                problems.append(
                    f"unknown parameter {path!r}{_did_you_mean(path, valid_paths)}"
                )
                continue
            for value, role in param.candidate_values():
                trial = base.model_copy(deep=True)
                try:
                    apply_overrides(trial, {path: value})
                except (ValidationError, ValueError) as exc:
                    problems.append(
                        f"{path} {role}={value!r} is rejected: {_reason(exc)}"
                    )
        if logged_metrics is not None and self.metric is not None:
            names = set(logged_metrics)
            if self.metric.name not in names:
                problems.append(
                    f"metric.name {self.metric.name!r} is not a logged metric"
                    f"{_did_you_mean(self.metric.name, names)}"
                )
        if problems:
            name = type(base).__name__
            header = f"Sweep incompatible with {name} — {len(problems)} problem(s):"
            raise ConfigError(
                header + "".join(f"\n  - {problem}" for problem in problems)
            )


def _leaves(
    parameters: dict[str, SweepParameter], prefix: str
) -> Iterator[tuple[str, SweepParameter]]:
    """Every swept parameter under ``parameters`` by dotted path, at any depth."""
    for name, param in parameters.items():
        if param.parameters is not None:
            yield from _leaves(param.parameters, f"{prefix}{name}.")
        else:
            yield f"{prefix}{name}", param


def _did_you_mean(name: str, options: Iterable[str]) -> str:
    match = difflib.get_close_matches(name, list(options), n=1)
    return f" — did you mean {match[0]!r}?" if match else ""


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            return str(errors[0].get("msg", exc))
    return str(exc).splitlines()[0]
