"""Export the config schema as JSON Schema for in-editor YAML autocomplete.

`Cfg.export_json_schema(path)` writes a Pydantic-derived JSON Schema and,
by default, merges a ``yaml.schemas`` mapping into ``.vscode/settings.json``
so any editor backed by ``yaml-language-server`` (VSCode + YAML extension,
neovim/helix with yamlls, ...) provides autocomplete, validation, and
inline docs while editing the matching YAML.

Run from the repo root:

    python examples/07_json_schema.py

That writes:

* ``examples/schemas/train.schema.json``     — the JSON Schema itself
* ``examples/.vscode/settings.json``         — the ``yaml.schemas`` mapping

To try it out, open the ``examples/`` directory in VSCode (with the
[YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
installed) and edit ``examples/configs/train.yaml`` — you should get
autocomplete for every overridable key, hover docs from the field
docstrings, and red squigglies for type errors. No per-file
``# yaml-language-server:`` modeline required.

Pass ``register=False`` to just emit the JSON (useful in CI); pass
``yaml_glob=`` to point the mapping at a custom file pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from onefig import ConfigModel


class OptimizerCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "adamw"
    """Which optimizer to use."""
    lr: float = 1e-4
    """Base learning rate."""
    weight_decay: float = 0.0
    """L2 regularization strength."""


class ModelCfg(ConfigModel):
    name: str
    """Identifier for the model (e.g. ``tiny-bert``)."""
    hidden_size: int = 768
    """Width of the hidden layers."""
    num_layers: int = 12
    """Depth of the transformer stack."""


class TrainCfg(ConfigModel):
    run_name: str
    """Human-readable label for this run."""
    epochs: int = 10
    """Number of training epochs."""
    model: ModelCfg
    optimizer: OptimizerCfg = OptimizerCfg()
    tags: list[str] = []
    """Free-form tags for experiment tracking."""


def main() -> None:
    examples_dir = Path(__file__).parent
    schema_path = examples_dir / "schemas" / "train.schema.json"
    settings_path = examples_dir / ".vscode" / "settings.json"

    TrainCfg.export_json_schema(
        schema_path,
        vscode_settings_path=settings_path,
    )

    print(f"Wrote schema:          {schema_path.relative_to(Path.cwd())}")
    print(f"Wrote VSCode mapping:  {settings_path.relative_to(Path.cwd())}")
    print()
    print("Next: open the `examples/` directory in VSCode (with the YAML")
    print("extension installed) and edit `configs/train.yaml`. You should")
    print("get autocomplete + validation + hover docs with no modeline.")


if __name__ == "__main__":
    main()
