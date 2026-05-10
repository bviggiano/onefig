"""Shell tab completion for ``key=value`` overrides.

Two flags drive the completion machinery:

* ``--onefig-install-completion <bash|zsh|fish>`` prints an install snippet
  for the calling script. Pipe it into a sourced rc file (or ``source``
  directly) to enable TAB completion for this script.
* ``--onefig-completions [PARTIAL]`` is the machine-readable callback used
  by the installed scripts. Each line of output is a completion candidate.

Try (from the repo root):

    python examples/06_completion.py --onefig-completions opt
    python examples/06_completion.py --onefig-install-completion bash

For TAB completion to actually trigger in your shell, the script must be
directly executable (e.g. ``chmod +x`` with a shebang line). Running via
``python script.py`` is not supported by the shell-completion mechanism.
"""

from __future__ import annotations

from typing import Literal

from onefig import ConfigModel


class OptimizerCfg(ConfigModel):
    kind: Literal["sgd", "adam", "adamw"] = "adamw"
    """Which optimizer to use."""
    lr: float = 1e-4
    """Base learning rate."""


class TrainCfg(ConfigModel):
    epochs: int = 10
    """Number of training epochs."""
    optimizer: OptimizerCfg = OptimizerCfg()


def main() -> None:
    cfg = TrainCfg()
    cfg.config_name = "train"
    cfg.update_from_cli()
    cfg.display()


if __name__ == "__main__":
    main()
