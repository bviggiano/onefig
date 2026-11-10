# Experiment tracking

## Snapshot the resolved config alongside the experiment

```python
cfg = TrainCfg.load("train")
cfg.update_from_cli()
cfg.freeze()

run_dir = Path("runs") / cfg.config_name
run_dir.mkdir(parents=True, exist_ok=True)
cfg.save_yaml(run_dir / "config.yaml")     # round-trippable via TrainCfg.load(...)
```

> Runnable demo: [`examples/07_freeze_and_snapshot.py`](https://github.com/bviggiano/onefig/blob/main/examples/07_freeze_and_snapshot.py).

## Name runs with `config_name`

`load` sets `cfg.config_name` to the YAML file's stem, so a config loaded from
`train.yaml` is named `"train"`. Use it for run directories, log lines, and
tracker run names. Pass `config_name=` to `load` to choose a different name, or
assign `cfg.config_name` while the config is unfrozen. Configs built directly
or with `from_dict` start with `None`.

```python
cfg = TrainCfg.load("configs/train.yaml")
cfg.config_name                # "train"
```

## Print the config as a tree

`cfg.display()` prints the resolved config as an ASCII tree, titled with
`config_name` (or `"Config"` when unset). `--show` on the command line prints
the same tree and exits.

```text
train:
├── epochs: 5
└── model:
    ├── name: 'tiny-bert'
    └── lr: 0.001
```

Pass `name=` to override the title. To print derived values that aren't config
fields below the tree, override `display_sections()` in your `ConfigModel`
subclass and return the extra blocks, or pass `sections=` at call time.

## Capture the running code's commit hash

Every config automatically captures the current `git HEAD` hash on
construction, available as `cfg.commit_hash`. Useful for tagging experiment
artifacts with the exact code version that produced them:

```python
cfg = TrainCfg.load("train")
print(cfg.commit_hash)         # "9f6e0438c2ea5ed..."
```

Best-effort and never raises; `cfg.commit_hash` is `None` when `git` isn't
available, the working directory isn't a repo, or capture otherwise fails. The
value is stored on a private attribute, so it stays out of `to_dict()`,
`to_flat_dict()`, and `save_yaml()` and won't pollute hyperparameter logs.

## Hyperparameter logging to W&B / MLflow

```python
import wandb
wandb.init(config=cfg.to_flat_dict())      # {"model.lr": 0.001, "epochs": 20, ...}
```

To restore the config later (e.g. for a re-run from a tracked run's params):

```python
cfg = TrainCfg.from_flat_dict(run.config)
```
