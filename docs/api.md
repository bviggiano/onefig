# API reference

```python
# Constructors
cfg = MyCfg.load(name_or_path)              # validate from YAML
cfg = MyCfg.from_dict({...})                # validate from nested dict
cfg = MyCfg.from_flat_dict({"a.b": 1})      # validate from flat dotted dict

# CLI overrides
cfg.update_from_cli(["lr=0.5"])              # key=value tokens (defaults to sys.argv[1:])
cfg.update_from_args(args)                   # pre-parsed argparse Namespace
cfg.update_from_env("MYAPP_")                # MYAPP_MODEL__LR=0.001 → cfg.model.lr

# Mutation control
cfg.freeze()                                 # recursive immutable mode
cfg.is_frozen                                # bool

# Identity
cfg.config_name                              # str | None — settable while unfrozen
cfg.commit_hash                              # str | None — git HEAD captured at construction

# Serialization
cfg.to_dict()                                # nested dict
cfg.to_flat_dict()                           # {"model.lr": 0.001, ...}
cfg.save_yaml("snapshot.yaml")              # write YAML to disk

# Diffing
cfg.diff(other_cfg_or_dict)                  # {"model.lr": (0.001, 0.01), ...}
cfg.diff_from_defaults()                     # diff vs type(cfg)()
cfg.print_diff(other_cfg_or_dict)            # aligned, color old → new
cfg.print_diff_from_defaults()               # same, against schema defaults
cfg.format_diff(other)                       # string form (for logging)
cfg.format_diff_from_defaults()              # string form

# Display
cfg.display(name="MyRun")                    # print ASCII tree to stdout
cfg.print_help()                             # schema-aware help (also via --help)
cfg.format_help()                            # same, returned as a string

# Shell completion
cfg.completion_candidates()                  # list of completion tokens
cfg.python_wrapper_completion_script("bash") # install snippet for `python <script>.py`
```

Tagged unions (see [Swappable components](unions.md)):

```python
from onefig import tagged_union

Store = tagged_union(LocalConfig, S3Config)             # tag key "name" by default
Store = tagged_union(LocalConfig, S3Config, tag="kind") # custom tag key
LocalConfig.name                                         # "local": derived, or an explicit ClassVar
```

The library installs a console command:

```bash
onefig install-python-completion <bash|zsh|fish>      # one-time, global
```
