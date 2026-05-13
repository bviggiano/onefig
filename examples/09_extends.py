"""Compose configs with a top-level ``extends:`` key.

A YAML file can declare ``extends: <path>`` (or a list of paths) to pull
in another file before validation. Parents are loaded first and the
current file is deep-merged on top, so you can keep a shared base
(``base.yaml``) and only spell out the deltas in each variant
(``small_fast.yaml``).

Run from the repo root:

    python examples/09_extends.py

Mechanics worth knowing:

* ``extends:`` accepts a string or a list. List form merges left-to-right
  (later parents override earlier ones); the current file always wins.
* Parents may themselves ``extends:`` something. Cycles are detected and
  raise.
* Paths resolve relative to the file containing ``extends:``. Absolute
  paths also work.
* ``${...}`` interpolation is resolved after the chain is merged, so
  parents can reference keys that only a child supplies.
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
    print("==> base.yaml")
    TrainCfg.load("examples/configs/base.yaml").display()

    print("\n==> small_fast.yaml  (extends: base.yaml, overrides epochs +")
    print("    model.num_layers, replaces tags; everything else inherited)")
    TrainCfg.load("examples/configs/small_fast.yaml").display()


if __name__ == "__main__":
    main()
