"""Shell tab completion for ``key=value`` overrides.

Three flags drive the completion machinery:

* ``--onefig-install-completion <bash|zsh|fish>`` prints an install snippet
  for the calling script. Used when the script is directly executable
  (``chmod +x`` with a shebang, or installed as a console-script).
* ``--onefig-install-python-completion <bash|zsh|fish>`` prints a generic
  install snippet that enables TAB completion for *every* onefig-based
  script invoked via ``python <script>.py``. One-time install per shell.
* ``--onefig-completions [PARTIAL]`` is the machine-readable callback used
  by both installed scripts. Each line of output is a completion candidate.

Try (from the repo root):

    python examples/06_completion.py --onefig-completions opt
    python examples/06_completion.py --onefig-install-completion bash
    python examples/06_completion.py --onefig-install-python-completion bash
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
