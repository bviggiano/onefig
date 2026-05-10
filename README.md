# onefig

[![PyPI](https://img.shields.io/pypi/v/onefig.svg)](https://pypi.org/project/onefig/)
[![Python versions](https://img.shields.io/pypi/pyversions/onefig.svg)](https://pypi.org/project/onefig/)
[![CI](https://github.com/bviggiano/onefig/actions/workflows/ci.yml/badge.svg)](https://github.com/bviggiano/onefig/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

```bash
pip install onefig
```
![hero.png](hero.png)

> **One config to rule them all!**

A simple, typed YAML config system for Python. Built on
[OmegaConf](https://omegaconf.readthedocs.io/) (loading + interpolation) and
[Pydantic v2](https://docs.pydantic.dev/) (typing + validation).

## Why

Hydra-style configs are powerful but heavyweight. Hand-rolled config classes are
brittle. onefig is the small, sharp middle: write a Pydantic schema, load YAML,
override from CLI, freeze it, and run.

> *"All we have to decide is what to do with the time that is given us."*

## Quickstart

> *"Speak, friend, and enter."*

Define your schema as a `ConfigModel`:

```python
from onefig import ConfigModel

class ModelCfg(ConfigModel):
    name: str
    lr: float = 1e-4

class TrainCfg(ConfigModel):
    epochs: int = 10
    model: ModelCfg
```

Author the YAML:

```yaml
# train.yaml
epochs: 5
model:
  name: tiny-bert
  lr: 0.001
```

Wire up a script:

```python
from onefig import ConfigModel

# (your schema as above)

def main():
    cfg = TrainCfg.load("train")          # name lookup, or pass a path
    cfg.update_from_cli()                  # picks up sys.argv[1:], supports --show
    cfg.freeze()
    cfg.display()
    train(cfg)

if __name__ == "__main__":
    main()
```

Run it:

```bash
python script.py                          # uses YAML defaults
python script.py lr=0.001 epochs=20       # CLI overrides
python script.py model.name=bert-base     # dotted-path override
python script.py lr=0.001 --show          # print resolved config and exit
```

## Common patterns

### CLI overrides without argparse

`update_from_cli` parses `key=value` tokens straight from `sys.argv`. No flag
declarations needed:

```python
cfg = TrainCfg.load("train")
cfg.update_from_cli()                      # python script.py lr=0.001 epochs=20
```

Leaf-key shortcuts work too — onefig resolves `lr` to `model.lr` if it's
unambiguous:

```bash
python script.py lr=0.001                  # → cfg.model.lr = 0.001
```

Need argparse instead (e.g. for help text, custom flag types, sweep tooling)?
Use `update_from_args` — same override engine, same leaf-key trick, takes a
parsed Namespace:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
args = parser.parse_args()

cfg = TrainCfg.load("train")
cfg.update_from_args(args)                 # None values are skipped by default
```

### Snapshot the resolved config alongside the experiment

> *"The road goes ever on and on"* — make sure you can find your way back.

```python
cfg = TrainCfg.load("train")
cfg.update_from_cli()
cfg.freeze()

run_dir = Path("runs") / cfg.config_name
run_dir.mkdir(parents=True, exist_ok=True)
cfg.save_yaml(run_dir / "config.yaml")     # round-trippable via TrainCfg.load(...)
```

### Capture the running code's commit hash

Every config automatically captures the current `git HEAD` hash on
construction, available as `cfg.commit_hash`. Useful for tagging experiment
artifacts with the exact code version that produced them:

```python
cfg = TrainCfg.load("train")
print(cfg.commit_hash)         # "9f6e0438c2ea5ed..."
```

Best-effort and never raises; `cfg.commit_hash` is `None` when `git` isn't
available, the working directory isn't a repo, or capture otherwise fails. The
value is stored on a private attribute, so it stays out of `to_dict()`,
`to_flat_dict()`, and `save_yaml()` and won't pollute hyperparameter logs.

### Hyperparameter logging to W&B / MLflow

```python
import wandb
wandb.init(config=cfg.to_flat_dict())      # {"model.lr": 0.001, "epochs": 20, ...}
```

To restore the config later (e.g. for a re-run from a tracked run's params):

```python
cfg = TrainCfg.from_flat_dict(run.config)
```

### Derived fields and final validation

For values that depend on other fields, use Pydantic's
[`model_post_init`](https://docs.pydantic.dev/latest/concepts/models/#custom-init)
hook. It fires after validation, so all fields are typed and present:

```python
class TrainCfg(ConfigModel):
    epochs: int
    steps_per_epoch: int
    warmup: int = 0
    total_steps: int = 0  # filled in below

    def model_post_init(self, _context) -> None:
        self.total_steps = self.epochs * self.steps_per_epoch + self.warmup
        if self.warmup > self.total_steps:
            raise ValueError("warmup cannot exceed total_steps")
```

Runs after every `load` / `from_dict` / `from_flat_dict` / direct construction.

### Cross-file YAML interpolation

```yaml
# train.yaml
run_name: ${model.name}-ep${epochs}
epochs: 5
model:
  name: tiny-bert
```

After loading, `cfg.run_name == "tiny-bert-ep5"`. Anything OmegaConf supports
(`${oc.env:HOME}`, `${some.other.field}`, etc.) works.

### Frozen configs

> *"...and in the darkness bind them."*

Once your config is finalized, freeze it so accidental mutation downstream
becomes a hard error:

```python
cfg.freeze()
cfg.epochs = 99            # raises FrozenConfigError
cfg.model.lr = 0.5         # also raises (freeze is recursive)
print(cfg.epochs)          # reads always work
```

## Features

- **YAML + interpolation** — `${other.key}`, `${oc.env:VAR}` resolved via OmegaConf.
- **Typed configs** — Pydantic validates on load and on every assignment;
  unknown fields rejected (`extra="forbid"`).
- **Two CLI override paths** — argparse-free `update_from_cli` for quick
  scripts, or `update_from_args` for custom argparse setups.
- **Smart leaf-key shortcuts** — `lr` resolves to `model.optimizer.lr` if it's
  unambiguous. Conflicts raise with a clear message.
- **Round-trippable** — `cfg.save_yaml()` ↔ `Cfg.load()`, and
  `cfg.to_flat_dict()` ↔ `Cfg.from_flat_dict()`.
- **Recursive freeze** — `cfg.freeze()` makes the whole tree immutable.
- **Tree display** — `cfg.display()` prints an ASCII tree, no `rich` dep.
- **Auto config name** — `cfg.config_name` is set from the YAML filename and
  available everywhere (run dirs, log lines, default tree titles).
- **Auto commit hash** — `cfg.commit_hash` captures the running code's
  `git HEAD` on construction (best-effort; `None` when unavailable).

## API reference

```python
# Constructors
cfg = MyCfg.load(name_or_path)              # validate from YAML
cfg = MyCfg.from_dict({...})                # validate from nested dict
cfg = MyCfg.from_flat_dict({"a.b": 1})      # validate from flat dotted dict

# CLI overrides
cfg.update_from_cli(["lr=0.5"])              # key=value tokens (defaults to sys.argv[1:])
cfg.update_from_args(args)                   # pre-parsed argparse Namespace

# Mutation control
cfg.freeze()                                 # recursive immutable mode
cfg.is_frozen                                # bool

# Identity
cfg.config_name                              # str | None — settable while unfrozen
cfg.commit_hash                              # str | None — git HEAD captured at construction

# Serialization
cfg.to_dict()                                # nested dict
cfg.to_flat_dict()                           # {"model.lr": 0.001, ...}
cfg.save_yaml("snapshot.yaml")              # write YAML to disk

# Display
cfg.display(name="MyRun")                    # print ASCII tree to stdout
```

## License

MIT

---

<p align="center">
  <img src="assets/lotr/gandalf-nod.gif" width="80" alt="Gandalf nod">
</p>
