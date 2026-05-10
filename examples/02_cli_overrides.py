"""CLI overrides via ``key=value`` tokens (no argparse).

Try a few of these from the repo root:

    python examples/02_cli_overrides.py
    python examples/02_cli_overrides.py epochs=20 lr=0.001
    python examples/02_cli_overrides.py model.hidden_size=1024 --show
    python examples/02_cli_overrides.py optimizer.kind=sgd

``lr`` resolves unambiguously to ``optimizer.lr`` thanks to leaf-key shortcuts.
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
    cfg.update_from_cli()  # consumes sys.argv[1:]
    cfg.freeze()
    cfg.display()


if __name__ == "__main__":
    main()
