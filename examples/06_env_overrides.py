"""Apply overrides from environment variables.

``update_from_env(prefix)`` reads every env var starting with ``prefix``,
strips it, splits the rest on ``__`` to address nested fields, and feeds
the result through the same override engine as the CLI path. Composes
naturally into a YAML → env → CLI precedence chain.

Run from the repo root:

    # Plain env override
    MYAPP_EPOCHS=20 python examples/06_env_overrides.py

    # Nested field via __
    MYAPP_MODEL__NAME=bert-large python examples/06_env_overrides.py

    # Leaf shortcut (resolves to model.name when unambiguous)
    MYAPP_NAME=bert-large python examples/06_env_overrides.py

    # YAML → env → CLI: CLI wins over env wins over YAML
    MYAPP_EPOCHS=20 python examples/06_env_overrides.py epochs=50

    # JSON-coerced values
    MYAPP_TAGS='["smoke", "fast"]' python examples/06_env_overrides.py
"""

from __future__ import annotations

from typing import Literal

from onefig import ConfigModel


class OptimizerCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0


class ModelCfg(ConfigModel):
    name: str
    hidden_size: int = 768
    num_layers: int = 12


class TrainCfg(ConfigModel):
    run_name: str
    epochs: int = 10
    model: ModelCfg
    optimizer: OptimizerCfg = OptimizerCfg()
    tags: list[str] = []


def main() -> None:
    cfg = TrainCfg.load("examples/configs/train.yaml")
    cfg.update_from_env("MYAPP_")     # env wins over YAML
    cfg.update_from_cli()             # CLI wins over env
    cfg.display()


if __name__ == "__main__":
    main()
