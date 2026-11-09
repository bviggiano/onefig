# onefig

A simple, typed YAML config system for Python that unites the best aspects of
[OmegaConf](https://omegaconf.readthedocs.io/) (flexible YAML loading and interpolation) and
[Pydantic v2](https://docs.pydantic.dev/) (strict typing and validation) in one API.

```bash
pip install onefig
```

## Quickstart

Define a `ConfigModel` which contains fields and their defaults. Using our built in parser, you can then directly override these from commandline as shown below:

```python
# script.py
from onefig import ConfigModel

class ModelCfg(ConfigModel):
    name: str = "tiny-bert"
    lr: float = 1e-4

class TrainCfg(ConfigModel):
    epochs: int = 10
    model: ModelCfg = ModelCfg()

def main():
    cfg = TrainCfg()
    cfg.update_from_cli()                  # picks up sys.argv[1:], supports --help / --show

if __name__ == "__main__":
    main()
```

Run it:

```bash
python script.py                          # uses schema defaults
python script.py lr=0.001 epochs=20       # CLI overrides
python script.py model.name=bert-base     # dotted-path override
python script.py lr=0.001 --show          # print resolved config and exit
python script.py --help                   # list every overridable field
```

> Runnable demo: [`examples/01_cli_overrides.py`](https://github.com/bviggiano/onefig/blob/main/examples/01_cli_overrides.py).

## Loading from YAML

When you want defaults to live alongside the code in a config file, swap the
direct constructor for `load`:

```yaml
# train.yaml
epochs: 5
model:
  name: tiny-bert
  lr: 0.001
```

```python
cfg = TrainCfg.load("train")              # name lookup, or pass a path
cfg.update_from_cli()                      # CLI still overrides
cfg.freeze()
```

YAML files support `${...}` interpolation and `extends:` composition; see
[YAML composition](yaml.md).

> Runnable demo: [`examples/02_basic.py`](https://github.com/bviggiano/onefig/blob/main/examples/02_basic.py).

## Next steps

- [Command-line overrides](cli.md): `key=value` overrides, shortcuts, argparse, and schema-aware `--help`.
- [Environment variables](env.md): overrides from `MYAPP_*`-style env vars.
- [Shell tab completion](completion.md): TAB completion of every overridable key.
- [YAML composition](yaml.md): `extends:` and cross-file interpolation.
- [Validation and freezing](validation.md): derived fields and immutable configs.
- [Experiment tracking](experiments.md): snapshots, commit hashes, and hyperparameter logging.
- [Diffing configs](diff.md): see exactly what changed between runs.
- [Integrations](integrations/index.md): typed configs for external tools such as W&B sweeps.
- [API reference](api.md): every public method at a glance.
