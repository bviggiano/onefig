# Diffing configs

`diff` reports the leaf-level changes between two configs (or between a
config and a flat/nested dict). Useful for PR-style "what changed in
this run" summaries, experiment-log entries, and confirming an
overridden run actually moved the fields you intended:

```python
baseline = TrainCfg.load("base")
run = TrainCfg.load("base")
run.update_from_cli(["lr=0.01", "epochs=20"])

run.diff(baseline)
# {"epochs": (20, 10), "model.lr": (0.01, 0.001)}
```

`diff_from_defaults` compares a config against a default-constructed
instance of its type — every entry is a field the user actually
deviated from:

```python
cfg.diff_from_defaults()
# {"model.lr": (1e-4, 0.001), "epochs": (10, 20)}
```

Keys present on only one side use `onefig.MISSING` as the absent
value, so diffing against partial dicts (or across schemas with
extra/missing keys) is well-defined. The result is an ordered dict
(self's keys first, in their declared order), so output is stable
across runs.

There are two human-readable views, one per use case:

**`print_diff(other)` — side-by-side comparison.** Every changed leaf
gets an `old → new` row, with red on the old side and green on the new.
Keys present on only one side render with a dimmed `<MISSING>`
placeholder:

```python
baseline.print_diff(run)
#   epochs            10           →  20
#   model.name        'tiny-bert'  →  'bert-large'
#   model.lr          0.0001       →  0.001

baseline.print_diff({"epochs": 99, "experiment.id": "abc123"})
#   epochs            10           →  99
#   model.name        'tiny-bert'  →  <MISSING>
#   experiment.id     <MISSING>    →  'abc123'
```

**`print_diff_from_defaults()` — config snapshot with override
highlights.** Every field in the config shows up. Fields you've
overridden render with red `default → current` (green); fields still
at their default render alone in green. Useful as a one-glance "what
does this run actually look like, and where did I deviate?":

```python
run.print_diff_from_defaults()
#   epochs                  10           →  20
#   debug                   False
#   model.name              'tiny-bert'  →  'bert-large'
#   model.hidden_size       768
#   model.lr                0.0001       →  0.001
```

Color is on by default when stdout is a tty; pass `color=False` to
force it off (or `color=True` to keep it on when piping to a file).

> Runnable demo: [`examples/08_diff.py`](https://github.com/bviggiano/onefig/blob/main/examples/08_diff.py).
