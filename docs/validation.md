# Validation and freezing

## Derived fields and final validation

For values that depend on other fields, use Pydantic's
[`model_post_init`](https://docs.pydantic.dev/latest/concepts/models/#custom-init)
hook. It fires after validation, so all fields are typed and present:

```python
class TrainCfg(ConfigModel):
    epochs: int
    steps_per_epoch: int
    warmup: int = 0
    total_steps: int = 0  # filled in below

    def model_post_init(self, _context) -> None:
        self.total_steps = self.epochs * self.steps_per_epoch + self.warmup
        if self.warmup > self.total_steps:
            raise ValueError("warmup cannot exceed total_steps")
```

Runs after every `load` / `from_dict` / `from_flat_dict` / direct construction.

## Frozen configs

Once your config is finalized, freeze it so accidental mutation downstream
becomes a hard error:

```python
cfg.freeze()
cfg.epochs = 99            # raises FrozenConfigError
cfg.model.lr = 0.5         # also raises (freeze is recursive)
print(cfg.epochs)          # reads always work
```

> Runnable demo: [`examples/07_freeze_and_snapshot.py`](https://github.com/bviggiano/onefig/blob/main/examples/07_freeze_and_snapshot.py).
