"""Visualize config diffs.

onefig ships two complementary visualizations, one per use case:

* ``cfg.print_diff(other)`` is a **side-by-side comparison**. Every
  changed leaf renders as ``old → new`` with red on the old side and
  green on the new. Keys present on only one side render with a dimmed
  ``<MISSING>`` placeholder.

* ``cfg.print_diff_from_defaults()`` is a **config snapshot with
  override highlights**. Every field shows up. Overridden fields render
  as ``default → current`` (red → green); fields still at their default
  render alone in green with no arrow. One-glance "what does this run
  look like, and where did I deviate?".

Both also have ``format_*`` siblings that return strings (useful for
logging or piping into a file). Color auto-detects via
``sys.stdout.isatty()`` and can be forced on or off with ``color=``.

Run from the repo root:

    python examples/08_diff.py
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


def _section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")  # bold


def main() -> None:
    baseline = TrainCfg()
    run = TrainCfg(
        epochs=20,
        model=ModelCfg(name="bert-large", num_layers=24),
        optimizer=OptimizerCfg(lr=1e-3),
    )

    # 1) Side-by-side comparison of two configs.
    _section("baseline.print_diff(run)  — side-by-side comparison")
    baseline.print_diff(run)

    # 2) Snapshot with override highlights — every field shown,
    #    overridden ones get the arrow.
    _section("run.print_diff_from_defaults()  — snapshot with overrides")
    run.print_diff_from_defaults()

    # 3) Cross-schema diff: partial dict. <MISSING> stands in for the
    #    absent side in either direction.
    partial = {"epochs": 99, "experiment.id": "abc123"}
    _section("baseline.print_diff(partial_dict)  — cross-schema with <MISSING>")
    baseline.print_diff(partial)

    # 4) Strings for logging instead of stdout.
    _section("run.format_diff_from_defaults()  — string form for logging")
    text = run.format_diff_from_defaults(color=False)  # plain text, no ANSI
    for line in text.split("\n"):
        print(f"  log| {line}")

    # 5) The raw diff dict is still available for programmatic use.
    _section("baseline.diff(run)  — raw dict for programmatic use")
    print(f"  {baseline.diff(run)}")


if __name__ == "__main__":
    main()
