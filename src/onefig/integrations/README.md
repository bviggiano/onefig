# Integrations

Typed onefig configs for the file **formats** of external tools.

Each integration models a third-party configuration format as a onefig `ConfigModel`, so you get the same
load-time validation and field-tree error reporting for a tool's config that you get for your own. Because only
the *format* is modeled — never the tool's Python package — importing an integration **adds no dependency**:
`onefig.integrations.wandb` works whether or not `wandb` is installed.

| Integration | Module | Models |
| --- | --- | --- |
| Weights & Biases | `onefig.integrations.wandb` | Sweep configuration |

More may be added over time (other experiment trackers, schedulers, etc.).

---

## <img src="wandb-logo.svg" alt="W&B" height="18"> Weights & Biases

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

`to_wandb()` returns the plain dict `wandb.sweep()` expects. Validating the swept parameter *paths and values*
against a specific onefig config schema (does `optimizer.lr` exist? is the value within its bounds?) is a
separate, optional layer built on top of this.
