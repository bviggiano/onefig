# YAML composition

## Compose configs with `extends:`

A top-level `extends:` key pulls in one or more parent YAML files before
validation. The parent is loaded first; the current file is deep-merged
on top, so you can keep a shared base and only spell out the deltas in
each variant:

```yaml
# base.yaml
epochs: 10
model:
  name: bert-base
  lr: 0.001
  arch:
    depth: 12
```

```yaml
# experiments/small-fast.yaml
extends: ../base.yaml
epochs: 3
model:
  name: tiny-bert
  arch:
    depth: 4         # overrides; `lr` is inherited from base
```

Result, after `TrainCfg.load("experiments/small-fast.yaml")`:

```python
{"epochs": 3, "model": {"name": "tiny-bert", "lr": 0.001,
                        "arch": {"depth": 4}}}
```

Mechanics:

- **List of parents** — `extends: [a.yaml, b.yaml]` merges left-to-right
  (later parents override earlier ones), then the current file overrides
  all parents.
- **Chains** — parents may themselves `extends:` something. Cycles are
  detected and raise.
- **Path resolution** — paths are relative to the file containing the
  `extends:` key. Absolute paths also work.
- **Interpolation timing** — `${...}` interpolations are resolved after
  the whole chain is merged, so a parent can reference a key that only
  a child supplies.

> Runnable demo: [`examples/09_extends.py`](https://github.com/bviggiano/onefig/blob/main/examples/09_extends.py).

## Cross-file YAML interpolation

```yaml
# train.yaml
run_name: ${model.name}-ep${epochs}
epochs: 5
model:
  name: tiny-bert
```

After loading, `cfg.run_name == "tiny-bert-ep5"`. Anything OmegaConf supports
(`${oc.env:HOME}`, `${some.other.field}`, etc.) works.
