# Weights & Biases

`WandbSweepConfig` models the [wandb sweep configuration
format](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration). wandb's sweep schema is precise and
easy to get subtly wrong — a typo'd key, a distribution missing a required argument, or a continuous range under
grid search will happily create a sweep that then wastes GPU hours. `WandbSweepConfig` catches all of that the
moment you load the file:

```python
from onefig.integrations.wandb import WandbSweepConfig

sweep = WandbSweepConfig.load("sweep.yaml")   # ConfigError (field tree) if the format is wrong
sweep_id = wandb.sweep(sweep.to_wandb(), project="my-project")
```

A malformed sweep is reported before it ever reaches wandb:

```
✗ Config Validation Error in 'sweep' — 1 problem found
└── parameters
    └── optimizer.lr  ✗ distribution 'log_uniform_values' requires `min` and `max` > 0
```

**What it validates:**

- **Unknown keys** — a mistyped `minn`/`parmeters` is rejected, not silently ignored.
- **Method** — one of `grid` / `random` / `bayes`; `bayes` requires a `metric`.
- **Grid vs. continuous** — `grid` search only accepts discrete `values` / `value`, not a range or distribution.
- **Distributions** — every distribution has its required arguments and nothing else: ranges need `min`/`max`
  (log-space ones must be `> 0`), `normal`/`log_normal` need `mu`/`sigma`, quantized `q_*` distributions need
  `q`, and `categorical` needs `values` (with matching `probabilities` that sum to 1).

`to_wandb()` returns the plain dict `wandb.sweep()` expects. A range keeps its bounds as written, so `{min: 8, max: 64}` reaches wandb as ints and is drawn as ints, as wandb does for a range of two ints.

## Validate against your config

The checks above are about the sweep *format*. You can go further and validate a sweep against a **specific**
config with `validate_against`, which applies each swept value to your config and runs its real validation:

```python
sweep.validate_against(MyRunConfig.load("base.yaml"), logged_metrics={"val/loss", "val/accuracy"})
```

It reports — all at once, before the sweep launches — a swept value outside a field's bounds, an invalid
`Literal` choice, a wrong type, a parameter *path* that doesn't exist (with a did-you-mean suggestion), and a
`metric.name` that isn't among your logged metrics:

```
✗ Sweep incompatible with RunConfig — 2 problem(s):
  - trainer.batch_size value=0 is rejected: Input should be greater than 0
  - unknown parameter 'optimzer.lr' — did you mean 'optimizer.lr'?
```

Because it routes values through the target config's own validators, it catches everything that config would
reject — bounds, enums, types, and cross-field constraints — with no rules duplicated here.

## List a sweep's trials

Grid and random sweeps are fully decided before any trial runs, so `trials()` lists their trials without wandb's
sweep server, as `{dotted path: value}` dicts:

```python
base = MyRunConfig.load("base.yaml")
for overrides in WandbSweepConfig.load("sweep.yaml").trials():
    train(base.replace(overrides))
```

- **Grid** runs every combination of each parameter's `values` (or `value`), in parameter-path order.
- **Random** draws `run_cap` trials, or `trials(count)`. A parameter is drawn as wandb's random search draws it:
  from `values` (weighted by `probabilities`), from a `min`/`max` range (ints when both bounds are ints), or from a
  `uniform`, `int_uniform` or `log_uniform_values` distribution.
- **Repeatable:** the draws are seeded by the sweep itself, so the same sweep always lists the same trials. Pass
  `seed=` for a different draw.
- **Nested** parameter groups are keyed by their dotted path, e.g. `optimizer.lr`.

Bayes search picks each trial from the results of the ones before it, and `early_terminate` stops trials by
comparing their results as they run, so a sweep using either raises `ConfigError`, as does a random sweep with no
count or a distribution `trials()` does not draw.
