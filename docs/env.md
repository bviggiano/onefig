# Environment variables

`update_from_env` reads overrides straight from the process environment,
using the same override engine as the CLI path (leaf and suffix-subpath
shortcuts, ambiguity detection, Pydantic re-validation):

```python
cfg = TrainCfg.load("train")
cfg.update_from_env("MYAPP_")              # MYAPP_EPOCHS=20 → cfg.epochs = 20
cfg.update_from_cli()                       # CLI wins over env
```

Nested fields are addressed with `__` (POSIX-legal substitute for `.`):

```bash
MYAPP_MODEL__LR=0.001 python script.py     # → cfg.model.lr = 0.001
MYAPP_LR=0.001        python script.py     # leaf shortcut → cfg.model.lr
```

Values are JSON-coerced (`true`, `5`, `[1,2]`, ...) before validation,
matching CLI behavior. Pass `case_sensitive=True`, a custom `delimiter`,
or `strict=False` if you need to deviate from the defaults; pass
`environ=` in tests to feed a synthetic mapping.

Compose freely into the precedence chain you want, e.g.
YAML → env → CLI:

```python
cfg = TrainCfg.load("train")
cfg.update_from_env("MYAPP_")
cfg.update_from_cli()
cfg.freeze()
```

> Runnable demo: [`examples/06_env_overrides.py`](https://github.com/bviggiano/onefig/blob/main/examples/06_env_overrides.py).
