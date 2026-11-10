"""Swap components with a tagged union.

A tagged union lets one field hold any of several variant configs. YAML, the
CLI and env vars pick the variant with its tag (``name``). The tag is a class
attribute, not a setting: it is derived from the class name (``MCARConfig`` ->
``mcar``) unless the class sets ``name: ClassVar[str]`` explicitly.

Run from the repo root:

    python examples/10_tagged_union.py
    python examples/10_tagged_union.py --help
    python examples/10_tagged_union.py missingness.name=informative --show
    python examples/10_tagged_union.py missingness.name=dropout sites=[a,b] --show
    python examples/10_tagged_union.py missingness.name=typo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Union

from onefig import ConfigModel, tagged_union


class InformativeConfig(ConfigModel):
    severity_slope: float = 0.8
    """How strongly the chance of a missing value rises with severity."""


class MCARConfig(ConfigModel):
    rate: float = 0.1
    """Probability that any value is missing, independent of everything else."""


class SiteDropout(ConfigModel):
    name: ClassVar[str] = "dropout"  # explicit; the derived tag is "site_dropout"
    sites: list[str] = []
    """Sites whose values are all missing."""


# A type checker can't follow the runtime-built union, so show it a plain one.
if TYPE_CHECKING:
    Missingness = Union[InformativeConfig, MCARConfig, SiteDropout]
else:
    Missingness = tagged_union(InformativeConfig, MCARConfig, SiteDropout)


class SiteConfig(ConfigModel):
    seed: int = 0
    missingness: Missingness = InformativeConfig()


def main() -> None:
    cfg = SiteConfig.load("examples/configs/site.yaml")  # selects mcar
    cfg.update_from_cli()  # e.g. missingness.name=informative
    cfg.display()

    variant = type(cfg.missingness)
    fields = list(variant.model_fields)
    print(f"\n{variant.__name__}: tag {variant.name!r}, fields {fields}")

    print("\nChanges from the defaults:")
    cfg.print_diff_from_defaults()


if __name__ == "__main__":
    main()
