# Integrations

Typed onefig configs for the file **formats** of external tools.

Each integration models a third-party configuration format as a onefig `ConfigModel`, so you get the same
load-time validation and field-tree error reporting for a tool's config that you get for your own. Because only
the *format* is modeled, never the tool's Python package, importing an integration **adds no dependency**:
`onefig.integrations.wandb` works whether or not `wandb` is installed.

<div class="integration-card" markdown>
<div class="integration-card__logo"><img src="https://raw.githubusercontent.com/bviggiano/onefig/main/assets/wandb-logo.svg" alt="Weights &amp; Biases" width="56" height="56"></div>
<div class="integration-card__body" markdown>

**[Weights & Biases](wandb.md)**

`WandbSweepConfig` validates a wandb sweep config the moment you load it: mistyped keys, malformed
distributions, a continuous range under grid search, and swept values your own config would reject are all
caught before the sweep launches.

`onefig.integrations.wandb` · [Learn more →](wandb.md)

</div>
</div>

More may be added over time (other experiment trackers, schedulers, etc.).
