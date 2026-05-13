"""Define a typed schema, load YAML, display the resolved config.

Run from the repo root:

    python examples/02_basic.py
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
    cfg.display()
    print()
    print(f"Resolved run_name (interpolated): {cfg.run_name!r}")
    print(f"Captured commit hash:             {cfg.commit_hash}")


if __name__ == "__main__":
    main()
