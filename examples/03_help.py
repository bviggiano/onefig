"""Schema-aware ``--help``: types, defaults, current values, and docstrings.

Try these from the repo root:

    python examples/03_help.py --help
    python examples/03_help.py epochs=20 --help     # --help short-circuits overrides
    python examples/03_help.py epochs=20 --show     # apply overrides, then print

Notice how ``Literal`` choices and the ``Loss`` enum members are surfaced
inline, and field docstrings (PEP 257) become the descriptions automatically.
``Field(description=...)`` takes precedence over the docstring when both are
present (see ``explicit`` below).
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import Field

from onefig import ConfigModel


class Loss(str, enum.Enum):
    CE = "cross_entropy"
    MSE = "mse"


class OptimizerCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "adamw"
    """Which optimizer to use."""
    lr: float = 1e-4
    """Base learning rate."""
    weight_decay: float = 0.0
    """L2 regularization strength."""


class ModelCfg(ConfigModel):
    name: str = "tiny-bert"
    """Pretrained model identifier."""
    hidden_size: int = 256
    """Width of the transformer hidden state."""


class TrainCfg(ConfigModel):
    epochs: int = 10
    """Number of training epochs."""
    loss: Loss = Loss.CE
    """Training loss function."""
    model: ModelCfg = ModelCfg()
    optimizer: OptimizerCfg = OptimizerCfg()
    checkpoint: str | None = None
    """Path to a checkpoint to resume from."""
    explicit: int = Field(default=7, description="Set via Field(description=...).")
    """This docstring is shadowed by the explicit Field description."""


def main() -> None:
    cfg = TrainCfg()
    cfg.config_name = "train"
    cfg.update_from_cli()
    cfg.display()


if __name__ == "__main__":
    main()
