# onefig

[![PyPI](https://img.shields.io/pypi/v/onefig.svg)](https://pypi.org/project/onefig/)
[![Downloads](https://static.pepy.tech/badge/onefig)](https://pepy.tech/project/onefig)
[![Python versions](https://img.shields.io/pypi/pyversions/onefig)](https://pypi.org/project/onefig/)
[![CI](https://github.com/bviggiano/onefig/actions/workflows/ci.yml/badge.svg)](https://github.com/bviggiano/onefig/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bviggiano.github.io/onefig/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

```bash
pip install onefig
```

![onefig](https://raw.githubusercontent.com/bviggiano/onefig/main/assets/hero.png)

A simple, typed YAML config system for Python that tries to unite the best aspects of various config libraries into one API.

> **One config to rule them all!**

## Why?

Configuration management is at the heart of most ML workflows, but can be wildly [annoying](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/configurator.py#L12-L14) to deal with. This repo is a simple pythonic config management system that tries to tie in the best aspects of [OmegaConf](https://omegaconf.readthedocs.io/) (flexible YAML loading and interpolation) and [Pydantic v2](https://docs.pydantic.dev/) (strict typing and validation), in a simple API. onefig takes heavy inspiration from [Hydra](https://github.com/facebookresearch/hydra), [Pydra](https://github.com/jordan-benjamin/pydra), and [tyro](https://github.com/brentyi/tyro).

`onefig` was built for my own projects, but figured others might find it useful too. Contributions and suggestions are welcome!

### Integrations

<a href="https://bviggiano.github.io/onefig/integrations/"><img src="https://raw.githubusercontent.com/bviggiano/onefig/main/assets/wandb-logo.svg" alt="Weights & Biases" height="22"></a>

`WandbSweepConfig` validates a wandb sweep config as soon as you load it, catching
bad distributions or out-of-bounds values before the sweep launches. See
the [integrations docs](https://bviggiano.github.io/onefig/integrations/).

## Quickstart

> *"Speak, friend, and enter."*

Define a `ConfigModel` which contains fields and their defaults. Using our built in parser, you can then directly override these from commandline as shown below:

```python
# script.py
from onefig import ConfigModel

class ModelCfg(ConfigModel):
    name: str = "tiny-bert"
    lr: float = 1e-4

class TrainCfg(ConfigModel):
    epochs: int = 10
    model: ModelCfg = ModelCfg()

def main():
    cfg = TrainCfg()
    cfg.update_from_cli()                  # picks up sys.argv[1:], supports --help / --show

if __name__ == "__main__":
    main()
```

Run it:

```bash
python script.py                          # uses schema defaults
python script.py lr=0.001 epochs=20       # CLI overrides
python script.py model.name=bert-base     # dotted-path override
python script.py lr=0.001 --show          # print resolved config and exit
python script.py --help                   # list every overridable field
```

> Runnable demo: [`examples/01_cli_overrides.py`](examples/01_cli_overrides.py).

### Loading from YAML

When you want defaults to live alongside the code in a config file, swap the
direct constructor for `load`:

```yaml
# train.yaml
epochs: 5
model:
  name: tiny-bert
  lr: 0.001
```

```python
cfg = TrainCfg.load("train")              # name lookup, or pass a path
cfg.update_from_cli()                      # CLI still overrides
cfg.freeze()
```

YAML files support `${...}` interpolation and `extends:` composition; see
[YAML composition](https://bviggiano.github.io/onefig/yaml/).

> Runnable demo: [`examples/02_basic.py`](examples/02_basic.py).


## Features

- [**YAML + interpolation**](https://bviggiano.github.io/onefig/yaml/#cross-file-yaml-interpolation) — `${other.key}`, `${oc.env:VAR}` resolved via OmegaConf.
- [**YAML composition**](https://bviggiano.github.io/onefig/yaml/) — top-level `extends: base.yaml` (or a list)
  deep-merges parent files into the current one before validation, with
  cycle detection and cross-file interpolation.
- [**Typed configs**](https://bviggiano.github.io/onefig/validation/) — Pydantic validates on load and on every assignment;
  unknown fields rejected (`extra="forbid"`).
- [**Two CLI override paths**](https://bviggiano.github.io/onefig/cli/) — argparse-free `update_from_cli` for quick
  scripts, or `update_from_args` for custom argparse setups.
- [**Env-var overrides**](https://bviggiano.github.io/onefig/env/) — `update_from_env("MYAPP_")` reads
  `MYAPP_MODEL__LR=0.001`-style env vars through the same override
  engine. Composes naturally with YAML → env → CLI precedence.
- [**Schema-aware `--help`**](https://bviggiano.github.io/onefig/cli/#schema-aware-help) — `python script.py --help` prints every
  overridable field with its type, default, current value, and docstring.
  `Literal` / `Enum` choices are surfaced inline.
- [**Shell tab completion**](https://bviggiano.github.io/onefig/completion/) — `onefig install-python-completion <shell>`
  enables TAB completion of every overridable key for any onefig script
  invoked via `python`. Supports `bash`, `zsh`, and `fish`.
- [**CLI / env override shortcuts**](https://bviggiano.github.io/onefig/cli/) — on the command line and in env vars,
  `lr` (or any trailing subpath like `model.lr`) resolves to its full path
  when unambiguous; ambiguous keys raise with a clear message. In Python
  code, always use the full attribute path.
- [**Round-trip serialization**](https://bviggiano.github.io/onefig/experiments/) — `cfg.save_yaml()` ↔ `Cfg.load()`, and
  `cfg.to_flat_dict()` ↔ `Cfg.from_flat_dict()`.
- [**Config diff**](https://bviggiano.github.io/onefig/diff/) — `cfg.diff(other)` and `cfg.diff_from_defaults()`
  surface leaf-level changes as `{path: (old, new)}`, with a `MISSING`
  sentinel for cross-schema gaps. Handy for run logs and PR-style
  "what changed" output.
- [**Swappable components**](https://bviggiano.github.io/onefig/unions/): `tagged_union(...)` lets a field hold any of several
  variant configs, picked by a `name` tag in YAML, CLI or env. The tag is a class
  attribute derived from the class name (`MCARConfig` → `mcar`), so variant fields
  are only real settings.
- [**Field markers**](https://bviggiano.github.io/onefig/markers/): attach your own typed objects to fields with
  `Field(default, *markers)`, find them anywhere in the tree with `cfg.find(Kind)`, and fill values in with `cfg.replace({...})`.
- [**Recursive freeze**](https://bviggiano.github.io/onefig/validation/#frozen-configs) — `cfg.freeze()` makes the whole tree immutable.
- [**Tree display**](https://bviggiano.github.io/onefig/experiments/#print-the-config-as-a-tree) — `cfg.display()` prints the config as an ASCII tree.
- [**Auto config name**](https://bviggiano.github.io/onefig/experiments/#name-runs-with-config_name) — `cfg.config_name` is set from the YAML filename and
  available everywhere (run dirs, log lines, default tree titles).
- [**Auto commit hash**](https://bviggiano.github.io/onefig/experiments/#capture-the-running-codes-commit-hash) — `cfg.commit_hash` captures the running code's
  `git HEAD` on construction (best-effort; `None` when unavailable).

## Documentation

Full documentation: **[bviggiano.github.io/onefig](https://bviggiano.github.io/onefig/)**

## License

MIT

<p align="center">
  <a href="https://www.youtube.com/watch?v=Y5NTgZA-xWE">
    <img src="https://raw.githubusercontent.com/bviggiano/onefig/main/assets/lotr/butmylord.gif" height="80" alt="But my lord, there is no Frodo">
  </a>
  &nbsp;&nbsp;
  <a href="https://youtube.com/shorts/u5sLlWVtC68">
    <img src="https://raw.githubusercontent.com/bviggiano/onefig/main/assets/lotr/gandalf-nod.gif" width="80" alt="Gandalf nod">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.youtube.com/watch?v=uE-1RPDqJAY">
    <img src="https://raw.githubusercontent.com/bviggiano/onefig/main/assets/lotr/theyre-taking-the-hobbits-to-isengard.gif" width="80" alt="They're taking the hobbits to Isengard">
  </a>
</p>
