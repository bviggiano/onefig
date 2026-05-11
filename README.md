# onefig

[![PyPI](https://img.shields.io/pypi/v/onefig.svg)](https://pypi.org/project/onefig/)
[![Python versions](https://img.shields.io/pypi/pyversions/onefig.svg)](https://pypi.org/project/onefig/)
[![CI](https://github.com/bviggiano/onefig/actions/workflows/ci.yml/badge.svg)](https://github.com/bviggiano/onefig/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Memes](https://img.shields.io/badge/lotr-memes-blueviolet)](#memes)

```bash
pip install onefig
```

> OPTIONAL: One-time TAB completion setup: see [Shell tab completion](#shell-tab-completion).

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

> Runnable demo: [`examples/01_basic.py`](examples/01_basic.py).

## Shell tab completion

> *"Not all those who wander are lost."*

Installing onefig adds an `onefig` console command with a one-time
install helper for shell tab completion. Follow the steps below to set
it up.

**1. Install onefig.**

```bash
pip install onefig
```

This registers the `onefig` console command on your `$PATH`.

**2. Append the snippet for your shell to its rc file.** This binds
completion once to `python` itself; every onefig-based script invoked
via `python script.py` then gets TAB completion automatically. Run
`echo $SHELL` if you're not sure which shell you're on.

```bash
# bash (Linux default, older macOS)
onefig install-python-completion bash >> ~/.bashrc
source ~/.bashrc
```

```bash
# zsh (macOS default since Catalina)
onefig install-python-completion zsh >> ~/.zshrc
source ~/.zshrc
```

```bash
# fish
onefig install-python-completion fish >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

> The shell arg passed to `onefig` *must* match the rc file you append
> to. The bash snippet uses bash builtins like `complete -F`, which
> aren't available in zsh; the zsh snippet uses `compdef` / `compadd`.

**3. Try it.**

```bash
python examples/06_completion.py opt<TAB>   # → optimizer.kind=  optimizer.lr=
python examples/06_completion.py l<TAB>     # → lr=
python examples/06_completion.py --<TAB>    # → --show  --help
```

> Runnable demo: [`examples/06_completion.py`](examples/06_completion.py).

**Preview before sourcing.** Both subcommands print their snippet to
stdout, so you can inspect what would be appended to your rc file, or
eval it for a one-shot test in the current shell only:

```bash
onefig install-python-completion bash           # print only
eval "$(onefig install-python-completion bash)" # one-shot in current shell
```

**How it works.** The completion list contains every overridable full
dotted path (suffixed with `=`), every unambiguous leaf-name shortcut,
and the special flags `--show`, `--help`, `-h`. Ambiguous leaves are
deliberately omitted so users aren't offered a shortcut the override
engine would refuse. The `python`-bound wrapper finds the first `.py`
argument on the command line, and before invoking the script it greps
the file for the literal word `onefig`. If the script doesn't reference
onefig, the wrapper returns silently — your TAB key is never going to
execute a random Python script with side effects at import time. For
onefig scripts, the wrapper invokes
`python <that script> --onefig-completions <prefix>` and uses the
output as the candidate list; the `--onefig-completions` flag is
intercepted inside `update_from_cli()` before `main()` runs, so even
the onefig script itself doesn't execute any of its own logic.

> If your script gets its `ConfigModel` through a re-export (e.g.
> `from mypkg.configs import TrainCfg`) and never mentions `onefig`
> directly, the grep gate will skip it. Add a `# onefig` comment
> anywhere in the file to opt in.

The same install snippet is also available as a flag on any onefig
script (``--onefig-install-python-completion <shell>``), which is
useful when the ``onefig`` command isn't on the user's ``$PATH`` (e.g.
inside an application virtualenv).

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

> Runnable demo: [`examples/03_help.py`](examples/03_help.py).

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

> Runnable demos: [`examples/02_cli_overrides.py`](examples/02_cli_overrides.py),
> [`examples/05_argparse.py`](examples/05_argparse.py).

### Environment variable overrides

> *"I will take the Ring, though I do not know the way."*

`update_from_env` reads overrides straight from the process environment,
using the same override engine as the CLI path (leaf-key shortcuts,
ambiguity detection, Pydantic re-validation):

```python
cfg = TrainCfg.load("train")
cfg.update_from_env("MYAPP_")              # MYAPP_EPOCHS=20 → cfg.epochs = 20
cfg.update_from_cli()                       # CLI wins over env
```

Nested fields are addressed with `__` (POSIX-legal substitute for `.`):

```bash
MYAPP_MODEL__LR=0.001 python script.py     # → cfg.model.lr = 0.001
MYAPP_LR=0.001        python script.py     # leaf shortcut → cfg.model.lr
```

Values are JSON-coerced (`true`, `5`, `[1,2]`, ...) before validation,
matching CLI behavior. Pass `case_sensitive=True`, a custom `delimiter`,
or `strict=False` if you need to deviate from the defaults; pass
`environ=` in tests to feed a synthetic mapping.

Compose freely into the precedence chain you want, e.g.
YAML → env → CLI:

```python
cfg = TrainCfg.load("train")
cfg.update_from_env("MYAPP_")
cfg.update_from_cli()
cfg.freeze()
```

> Runnable demo: [`examples/07_env_overrides.py`](examples/07_env_overrides.py).

### Snapshot the resolved config alongside the experiment

> *"The road goes ever on and on"*

```python
cfg = TrainCfg.load("train")
cfg.update_from_cli()
cfg.freeze()

run_dir = Path("runs") / cfg.config_name
run_dir.mkdir(parents=True, exist_ok=True)
cfg.save_yaml(run_dir / "config.yaml")     # round-trippable via TrainCfg.load(...)
```

> Runnable demo: [`examples/04_freeze_and_snapshot.py`](examples/04_freeze_and_snapshot.py).

### Diff configs

> *"The Council of Elrond."*

`diff` reports the leaf-level changes between two configs (or between a
config and a flat/nested dict). Useful for PR-style "what changed in
this run" summaries, experiment-log entries, and confirming an
overridden run actually moved the fields you intended:

```python
baseline = TrainCfg.load("base")
run = TrainCfg.load("base")
run.update_from_cli(["lr=0.01", "epochs=20"])

run.diff(baseline)
# {"epochs": (20, 10), "model.lr": (0.01, 0.001)}
```

`diff_from_defaults` compares a config against a default-constructed
instance of its type — every entry is a field the user actually
deviated from:

```python
cfg.diff_from_defaults()
# {"model.lr": (1e-4, 0.001), "epochs": (10, 20)}
```

Keys present on only one side use `onefig.MISSING` as the absent
value, so diffing against partial dicts (or across schemas with
extra/missing keys) is well-defined. The result is an ordered dict
(self's keys first, in their declared order), so output is stable
across runs.

There are two human-readable views, one per use case:

**`print_diff(other)` — side-by-side comparison.** Every changed leaf
gets an `old → new` row, with red on the old side and green on the new.
Keys present on only one side render with a dimmed `<MISSING>`
placeholder:

```python
baseline.print_diff(run)
#   epochs            10           →  20
#   model.name        'tiny-bert'  →  'bert-large'
#   model.lr          0.0001       →  0.001

baseline.print_diff({"epochs": 99, "experiment.id": "abc123"})
#   epochs            10           →  99
#   model.name        'tiny-bert'  →  <MISSING>
#   experiment.id     <MISSING>    →  'abc123'
```

**`print_diff_from_defaults()` — config snapshot with override
highlights.** Every field in the config shows up. Fields you've
overridden render with red `default → current` (green); fields still
at their default render alone in green. Useful as a one-glance "what
does this run actually look like, and where did I deviate?":

```python
run.print_diff_from_defaults()
#   epochs                  10           →  20
#   debug                   False
#   model.name              'tiny-bert'  →  'bert-large'
#   model.hidden_size       768
#   model.lr                0.0001       →  0.001
```

Color is on by default when stdout is a tty; pass `color=False` to
force it off (or `color=True` to keep it on when piping to a file).

> Runnable demo: [`examples/09_diff.py`](examples/09_diff.py).

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

### Compose configs with `extends:`

> *"Many that live deserve death. And some that die deserve life. Can you give it to them, Frodo?"*

A top-level `extends:` key pulls in one or more parent YAML files before
validation. The parent is loaded first; the current file is deep-merged
on top, so you can keep a shared base and only spell out the deltas in
each variant:

```yaml
# base.yaml
epochs: 10
model:
  name: bert-base
  lr: 0.001
  arch:
    depth: 12
```

```yaml
# experiments/small-fast.yaml
extends: ../base.yaml
epochs: 3
model:
  name: tiny-bert
  arch:
    depth: 4         # overrides; `lr` is inherited from base
```

Result, after `TrainCfg.load("experiments/small-fast.yaml")`:

```python
{"epochs": 3, "model": {"name": "tiny-bert", "lr": 0.001,
                        "arch": {"depth": 4}}}
```

Mechanics:

- **List of parents** — `extends: [a.yaml, b.yaml]` merges left-to-right
  (later parents override earlier ones), then the current file overrides
  all parents.
- **Chains** — parents may themselves `extends:` something. Cycles are
  detected and raise.
- **Path resolution** — paths are relative to the file containing the
  `extends:` key. Absolute paths also work.
- **Interpolation timing** — `${...}` interpolations are resolved after
  the whole chain is merged, so a parent can reference a key that only
  a child supplies.

> Runnable demo: [`examples/08_extends.py`](examples/08_extends.py).

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

> Runnable demo: [`examples/04_freeze_and_snapshot.py`](examples/04_freeze_and_snapshot.py).

## Features

- **YAML + interpolation** — `${other.key}`, `${oc.env:VAR}` resolved via OmegaConf.
- **YAML composition** — top-level `extends: base.yaml` (or a list)
  deep-merges parent files into the current one before validation, with
  cycle detection and cross-file interpolation.
- **Typed configs** — Pydantic validates on load and on every assignment;
  unknown fields rejected (`extra="forbid"`).
- **Two CLI override paths** — argparse-free `update_from_cli` for quick
  scripts, or `update_from_args` for custom argparse setups.
- **Env-var overrides** — `update_from_env("MYAPP_")` reads
  `MYAPP_MODEL__LR=0.001`-style env vars through the same override
  engine. Composes naturally with YAML → env → CLI precedence.
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
- **Config diff** — `cfg.diff(other)` and `cfg.diff_from_defaults()`
  surface leaf-level changes as `{path: (old, new)}`, with a `MISSING`
  sentinel for cross-schema gaps. Handy for run logs and PR-style
  "what changed" output.
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
cfg.update_from_env("MYAPP_")                # MYAPP_MODEL__LR=0.001 → cfg.model.lr

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

# Diffing
cfg.diff(other_cfg_or_dict)                  # {"model.lr": (0.001, 0.01), ...}
cfg.diff_from_defaults()                     # diff vs type(cfg)()
cfg.print_diff(other_cfg_or_dict)            # aligned, color old → new
cfg.print_diff_from_defaults()               # same, against schema defaults
cfg.format_diff(other)                       # string form (for logging)
cfg.format_diff_from_defaults()              # string form


# Display
cfg.display(name="MyRun")                    # print ASCII tree to stdout
cfg.print_help()                             # schema-aware help (also via --help)
cfg.format_help()                            # same, returned as a string

# Shell completion
cfg.completion_candidates()                  # list of completion tokens
cfg.python_wrapper_completion_script("bash") # install snippet for `python <script>.py`
```

The library installs a console command:

```bash
onefig install-python-completion <bash|zsh|fish>      # one-time, global
```

## Examples

Runnable scripts live in [`examples/`](examples/), with one Python file per
feature plus a notebook walkthrough. Start with
[`examples/01_basic.py`](examples/01_basic.py); the complete list is in
[`examples/README.md`](examples/README.md).

## License

MIT

---

## Memes

<p align="center">
  <a href="https://www.youtube.com/watch?v=Y5NTgZA-xWE">
    <img src="assets/lotr/one-ring.png" width="80" alt="The One Ring">
  </a>
  &nbsp;&nbsp;
  <a href="https://youtube.com/shorts/u5sLlWVtC68">
    <img src="assets/lotr/gandalf-nod.gif" width="80" alt="Gandalf nod">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.youtube.com/watch?v=uE-1RPDqJAY">
    <img src="assets/lotr/theyre-taking-the-hobbits-to-isengard.gif" width="80" alt="They're taking the hobbits to Isengard">
  </a>
</p>
