"""Argparse-flavored overrides via ``update_from_args``.

Use this when you want full control over flag definitions, custom help text,
or when integrating with sweep tooling that expects a real argparse parser.

Try:

    python examples/05_argparse.py --help
    python examples/05_argparse.py --epochs 20 --lr 0.001
"""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model with onefig.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--hidden-size",
        dest="model.hidden_size",
        type=int,
        default=None,
        help="Hidden size (mapped to model.hidden_size).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainCfg.load("examples/configs/train.yaml")
    cfg.update_from_args(args)  # None values are skipped by default
    cfg.freeze()
    cfg.display()


if __name__ == "__main__":
    main()
