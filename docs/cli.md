# Command-line overrides

## CLI overrides without argparse

`update_from_cli` parses `key=value` tokens directly from `sys.argv`, so no
flag declarations are required:

```python
cfg = TrainCfg.load("train")
cfg.update_from_cli()                      # python script.py lr=0.001 epochs=20
```

Leaf and suffix-subpath shortcuts are also supported on the command line:
onefig resolves `lr` (or any trailing subpath like `model.lr`) to the full
path when it is unambiguous in the schema — a base-level field wins over a
nested one, and an ambiguous key fails loudly:

```bash
python script.py lr=0.001                  # → cfg.model.lr = 0.001
```

> **Shortcuts are for CLI and env overrides only.** In Python code, always use
> the full attribute path — `cfg.model.lr`, not `cfg.lr` (which raises).

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

> Runnable demos: [`examples/01_cli_overrides.py`](https://github.com/bviggiano/onefig/blob/main/examples/01_cli_overrides.py),
> [`examples/05_argparse.py`](https://github.com/bviggiano/onefig/blob/main/examples/05_argparse.py).

## Schema-aware `--help`

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
from pydantic import Field

# Attach help text with Pydantic's Field(description=...).
class OptCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = Field("sgd", description="Which optimizer to use.")
    lr: float = Field(1e-4, description="Learning rate.")

class TrainCfg(ConfigModel):
    epochs: int = Field(10, description="Number of epochs.")
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
discover valid values from the help output. Descriptions come from
`Field(description=...)`; PEP 257 field docstrings (a `"""..."""` block
on the line *below* the field) are also supported, with
`Field(description=...)` taking precedence when both are provided. The
same output is available as `cfg.print_help()` (or `cfg.format_help()` for
the string), which is useful when integrating with an argparse-driven
entry point.

> Runnable demo: [`examples/04_help.py`](https://github.com/bviggiano/onefig/blob/main/examples/04_help.py).
