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

A simple, typed YAML config system for Python that tries to unite the best aspects of various config libaries into one API.

> **One config to rule them all!**

## Why?

Configuration management is at the heart of most ML workflows, but can be wildly [annoying](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/configurator.py#L12-L14) to deal with. This repo is a simple pythonic config management system that tries to tie in the best aspects of [OmegaConf](https://omegaconf.readthedocs.io/) (flexible YAML loading and interpolation) and [Pydantic v2](https://docs.pydantic.dev/) (strict typing and validation), in a simple API. onefig takes heavy inspiration from [Hydra](https://github.com/facebookresearch/hydra), [Pydra](https://github.com/jordan-benjamin/pydra), and [tyro](https://github.com/brentyi/tyro).

`onefig` was built for my own projects, but figured others might find it useful too. Contributions and suggestions are welcome!

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

Use the config in a script:

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
python script.py --help                   # list every overridable field
```

## Common patterns

> *"All we have to decide is what to do with the time that is given us."*

### Schema-aware `--help`

`update_from_cli` intercepts `--help` / `-h` and prints every overridable
field (with its type, default, current value, and docstring) before exiting.
The help text is generated directly from the `ConfigModel` schema, so it
stays accurate without manual maintenance. Each nested `ConfigModel`
renders as its own sub-panel. Entries show the leaf name alone when that
name resolves to a single field, and the full dotted path otherwise (which
matches the only form the override engine accepts when a leaf is
ambiguous):

```python
from typing import Literal

class OptCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "sgd"
    """Which optimizer to use."""
    lr: float = 1e-4
    """Learning rate."""

class TrainCfg(ConfigModel):
    epochs: int = 10
    """Number of epochs."""
    optimizer: OptCfg = OptCfg()
```

```text
$ python script.py --help
╭─ train ──────────────────────────────────────────────────────────────────────────────╮
│                                                                                      │
│ Override fields with key=value (or use --show / --help).                             │
│                                                                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   epochs : int = 10                                                                  │
│       Number of epochs.                                                              │
│                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯

╭─ optimizer ──────────────────────────────────────────────────────────────────────────╮
│                                                                                      │
│   kind : Literal['sgd', 'adam', 'adamw'] = 'sgd'                                     │
│       Which optimizer to use.                                                        │
│                                                                                      │
│   lr : float = 0.0001                                                                │
│       Learning rate.                                                                 │
│                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯

╭─ flags ──────────────────────────────────────────────────────────────────────────────╮
│                                                                                      │
│ --show         Print the resolved config and exit.                                   │
│ --help, -h     Show this help and exit.                                              │
│                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

Each entry reads like a Python annotation: `<label> : <type> = <current value>`
on the first line, with the description hang-indented below. When the current
value differs from the declared default, `(default: X)` is appended to the
description (it wraps to the next indented line if it overflows). Run with
`lr=0.5` and the `lr` block becomes:

```text
│   lr : float = 0.5                                                                   │
│       Learning rate. (default: 0.0001)                                               │
```

`Literal[...]` choices and `Enum` members are listed inline so users can
discover valid values from the help output. Field docstrings (PEP 257) are
used as descriptions automatically; an explicit `Field(description=...)`
takes precedence when both are provided. The same output is available as
`cfg.print_help()` (or `cfg.format_help()` for the string), which is
useful when integrating with an argparse-driven entry point.

### Shell tab completion

Installing onefig adds an `onefig` console command with two install
helpers for shell tab completion. The recommended one-time setup hooks
completion onto `python` itself, so every onefig-based script invoked
via `python script.py` gets completion automatically:

```bash
# One-time install for the current shell.
onefig install-python-completion bash >> ~/.bashrc
source ~/.bashrc

# Then TAB completion works on any onefig script invoked via python.
python train.py opt<TAB>          # → optimizer.kind=  optimizer.lr=  ...
python train.py l<TAB>            # → lr=
```

The completion list contains every overridable full dotted path (suffixed
with `=`), every unambiguous leaf-name shortcut, and the special flags
`--show`, `--help`, `-h`. Ambiguous leaves are deliberately omitted so
users aren't offered a shortcut the override engine would refuse. The
wrapper finds the first `.py` argument on the command line, invokes
`python <that script> --onefig-completions <prefix>`, and uses the
output as the candidate list. Non-onefig scripts produce no candidates,
and the user falls back to the shell's default behavior.

If your script is directly executable (a shebang plus `chmod +x`, or a
console-script entry point), you can bind completion to its command name
instead, which avoids re-walking the command line on every TAB:

```bash
onefig install-completion bash --prog train.py >> ~/.bashrc
source ~/.bashrc

./train.py opt<TAB>          # → optimizer.kind=  optimizer.lr=  ...
```

Both helpers also accept `zsh` and `fish` as the target shell.

The same install snippets are also available as flags on any onefig
script (``--onefig-install-python-completion <shell>`` and
``--onefig-install-completion <shell>``), which is useful when the
``onefig`` command isn't on the user's ``$PATH`` (e.g. inside an
application virtualenv).

### CLI overrides without argparse

`update_from_cli` parses `key=value` tokens directly from `sys.argv`, so no
flag declarations are required:

```python
cfg = TrainCfg.load("train")
cfg.update_from_cli()                      # python script.py lr=0.001 epochs=20
```

Leaf-key shortcuts are also supported: onefig resolves `lr` to `model.lr`
when the leaf name is unambiguous in the schema:

```bash
python script.py lr=0.001                  # → cfg.model.lr = 0.001
```

For argparse-driven setups (custom flag types, integration with sweep
tooling, etc.), `update_from_args` accepts a parsed `Namespace` and uses
the same override engine and leaf-key resolution:

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
- **Schema-aware `--help`** — `python script.py --help` prints every
  overridable field with its type, default, current value, and docstring.
  `Literal` / `Enum` choices are surfaced inline.
- **Shell tab completion** — `onefig install-python-completion <shell>`
  enables TAB completion of every overridable key for any onefig script
  invoked via `python`. Supports `bash`, `zsh`, and `fish`.
- **Leaf-key shortcuts** — `lr` resolves to `model.optimizer.lr` when the
  leaf name is unambiguous; conflicts raise with a clear message.
- **Round-trip serialization** — `cfg.save_yaml()` ↔ `Cfg.load()`, and
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
cfg.print_help()                             # schema-aware help (also via --help)
cfg.format_help()                            # same, returned as a string

# Shell completion
cfg.completion_candidates()                  # list of completion tokens
cfg.shell_completion_script("bash")          # install snippet for the calling script
cfg.python_wrapper_completion_script("bash") # install snippet for `python <script>.py`
```

The library installs a console command:

```bash
onefig install-python-completion <bash|zsh|fish>      # one-time, global
onefig install-completion <bash|zsh|fish> --prog NAME # per-script
```

## Examples

Runnable scripts live in [`examples/`](examples/), with one Python file per
feature plus a notebook walkthrough. Start with
[`examples/01_basic.py`](examples/01_basic.py); the complete list is in
[`examples/README.md`](examples/README.md).

## License

MIT

---

<p align="center">
  <img src="assets/lotr/gandalf-nod.gif" width="80" alt="Gandalf nod">
</p>
