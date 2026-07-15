from __future__ import annotations

from onefig._diff import MISSING
from onefig._format import flatten, format_tree, unflatten
from onefig._union import tagged_union
from onefig.model import ConfigModel

__all__ = [
    "ConfigModel",
    "MISSING",
    "flatten",
    "format_tree",
    "tagged_union",
    "unflatten",
]
__version__ = "0.1.1"
