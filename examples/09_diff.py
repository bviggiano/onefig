"""Diff configs to surface what actually changed.

``cfg.diff(other)`` returns ``{dotted_path: (self_value, other_value)}``
for every leaf that differs. ``cfg.diff_from_defaults()`` compares the
current config against a default-constructed instance of its type.
Useful for run logs and PR-style "what changed in this run" output.

Run from the repo root:

    python examples/09_diff.py
"""

from __future__ import annotations

from typing import Literal

from onefig import ConfigModel


class OptimizerCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0


class ModelCfg(ConfigModel):
    name: str = "tiny-bert"
    hidden_size: int = 768
    num_layers: int = 12


class TrainCfg(ConfigModel):
    epochs: int = 10
    model: ModelCfg = ModelCfg()
    optimizer: OptimizerCfg = OptimizerCfg()
    tags: list[str] = []


def main() -> None:
    baseline = TrainCfg()
    run = TrainCfg(
        epochs=20,
        model=ModelCfg(name="bert-large", num_layers=24),
        optimizer=OptimizerCfg(lr=1e-3),
    )

    # 1) Visualize what `run` changed vs. baseline. Call it on `baseline`
    #    so the arrow reads "baseline → run" — i.e. "this is what changed".
    print("baseline.print_diff(run):")
    baseline.print_diff(run)

    # 2) Same idea framed against schema defaults — single-call form.
    print("\nrun.print_diff_from_defaults():")
    run.print_diff_from_defaults()

    # 3) Cross-schema diff: dict misses a key the config has, and adds one
    #    the config doesn't. <MISSING> stands in for the absent side.
    partial = {"epochs": 99, "experiment.id": "abc123"}
    print("\nbaseline.print_diff(partial_dict):")
    baseline.print_diff(partial)

    # 4) Programmatic access: the raw dict is still what diff() returns.
    print("\nbaseline.diff(run) as a plain dict:")
    print(f"  {baseline.diff(run)}")


if __name__ == "__main__":
    main()
