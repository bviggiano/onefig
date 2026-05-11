# Examples

Runnable scripts that demonstrate onefig's main features. Run them from the
repo root so the relative paths to `examples/configs/train.yaml` resolve.

| Script | Demonstrates |
| ------ | ------------ |
| [`01_basic.py`](01_basic.py) | Defining a typed schema, loading YAML, OmegaConf interpolation, `display()`. |
| [`02_cli_overrides.py`](02_cli_overrides.py) | `key=value` overrides via `update_from_cli`, leaf-key shortcuts, `--show`. |
| [`03_help.py`](03_help.py) | Schema-aware `--help`: types, defaults, `Literal` / `Enum` choices, docstrings. |
| [`04_freeze_and_snapshot.py`](04_freeze_and_snapshot.py) | `freeze()`, `save_yaml()`, `commit_hash`. |
| [`05_argparse.py`](05_argparse.py) | Argparse-flavored overrides via `update_from_args`. |
| [`06_completion.py`](06_completion.py) | Shell tab completion for `bash`, `zsh`, and `fish`. |
| [`07_env_overrides.py`](07_env_overrides.py) | Environment variable overrides via `update_from_env`. |
| [`08_extends.py`](08_extends.py) | YAML composition via top-level `extends:` (Hydra-lite). |
| [`notebook.ipynb`](notebook.ipynb) | A guided tour of the same APIs in notebook form. |

Quick try:

```bash
python examples/01_basic.py
python examples/02_cli_overrides.py epochs=20 lr=0.001 --show
python examples/03_help.py --help
python examples/04_freeze_and_snapshot.py epochs=3 lr=0.01
python examples/05_argparse.py --epochs 20 --lr 0.001
python examples/06_completion.py --onefig-completions opt
MYAPP_EPOCHS=20 MYAPP_MODEL__NAME=bert python examples/07_env_overrides.py
python examples/08_extends.py
```
