"""Optional integrations with external tools, modeled as typed onefig configs.

Each integration models a third-party *format* (not the tool itself), so importing
it never requires the tool to be installed. Currently:
:mod:`onefig.integrations.wandb` (Weights & Biases sweep configs).
"""

from __future__ import annotations

from onefig.integrations.wandb import WandbSweepConfig

__all__ = ["WandbSweepConfig"]
