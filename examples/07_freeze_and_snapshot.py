"""Freeze the resolved config and snapshot it next to your run artifacts.

Run from the repo root:

    python examples/07_freeze_and_snapshot.py epochs=3 lr=0.01

The script writes ``runs/<config_name>/config.yaml`` so you can reproduce the
run by loading the snapshot back through ``TrainCfg.load(...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from onefig import ConfigModel
from onefig.model import FrozenConfigError


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
    cfg.update_from_cli()
    cfg.freeze()

    run_dir = Path("runs") / (cfg.config_name or "default")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.yaml"
    cfg.save_yaml(snapshot)

    cfg.display()
    print(f"\nSnapshot written to: {snapshot}")
    print(f"Captured commit hash: {cfg.commit_hash}")

    try:
        cfg.epochs = 99
    except FrozenConfigError as e:
        print(f"\nMutation refused: {e}")


if __name__ == "__main__":
    main()
