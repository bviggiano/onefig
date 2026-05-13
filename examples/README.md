# Examples

Runnable scripts that demonstrate onefig's main features. Run them from the
repo root so the relative paths to `examples/configs/train.yaml` resolve.

| Script | Demonstrates |
| ------ | ------------ |
| [`01_cli_overrides.py`](01_cli_overrides.py) | `key=value` overrides via `update_from_cli`, leaf-key shortcuts, `--show`. |
| [`02_basic.py`](02_basic.py) | Defining a typed schema, loading YAML, OmegaConf interpolation, `display()`. |
| [`03_completion.py`](03_completion.py) | Shell tab completion for `bash`, `zsh`, and `fish`. |
| [`04_help.py`](04_help.py) | Schema-aware `--help`: types, defaults, `Literal` / `Enum` choices, docstrings. |
| [`05_argparse.py`](05_argparse.py) | Argparse-flavored overrides via `update_from_args`. |
| [`06_env_overrides.py`](06_env_overrides.py) | Environment variable overrides via `update_from_env`. |
| [`07_freeze_and_snapshot.py`](07_freeze_and_snapshot.py) | `freeze()`, `save_yaml()`, `commit_hash`. |
| [`08_diff.py`](08_diff.py) | `diff()` / `diff_from_defaults()` plus colored CLI visualization. |
| [`09_extends.py`](09_extends.py) | YAML composition via top-level `extends:` (Hydra-lite). |
| [`notebook.ipynb`](notebook.ipynb) | A guided tour of the same APIs in notebook form. |

Quick try:

```bash
python examples/01_cli_overrides.py epochs=20 lr=0.001 --show
python examples/02_basic.py
python examples/03_completion.py --onefig-completions opt
python examples/04_help.py --help
python examples/05_argparse.py --epochs 20 --lr 0.001
MYAPP_EPOCHS=20 MYAPP_MODEL__NAME=bert python examples/06_env_overrides.py
python examples/07_freeze_and_snapshot.py epochs=3 lr=0.01
python examples/08_diff.py
python examples/09_extends.py
```
