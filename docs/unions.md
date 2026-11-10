# Swappable components

A **tagged union** lets one field hold any of several config classes, called
*variants*. YAML, the CLI and env vars choose the variant with a *tag* key,
`name` by default, and onefig validates the remaining keys against that
variant alone. Use one wherever a component should be swappable: an optimizer,
a storage backend, a data source, a reporting convention.

```python
from onefig import ConfigModel, tagged_union


class InformativeConfig(ConfigModel):
    severity_slope: float = 0.8


class MCARConfig(ConfigModel):
    rate: float = 0.1


Missingness = tagged_union(InformativeConfig, MCARConfig)


class SiteConfig(ConfigModel):
    missingness: Missingness = InformativeConfig()
```

```yaml
# site.yaml
missingness:
  name: mcar
  rate: 0.2
```

`SiteConfig.load("site.yaml").missingness` is an `MCARConfig` with `rate=0.2`.

## The tag is a class attribute, not a setting

Each variant's tag lives on the class as a `ClassVar`. A variant's fields are
therefore only its real settings:

- `MCARConfig.model_fields` is `{"rate": ...}`. `--help`, tab completion and
  `MCARConfig(...)` see `rate` and nothing else, and `MCARConfig(name="mcar")`
  is rejected.
- `cfg.missingness.name` reads the tag (`"mcar"`), from the class.
- Every dump writes the tag back next to the fields: `to_dict()`,
  `to_flat_dict()` (`missingness.name`), `save_yaml()`, `diff()` and
  `display()`. A saved config records which variant it used and loads back to
  the same one.
- `name` is reserved inside a variant's YAML: it always selects the variant.

## Where the tag comes from

A variant without an explicit tag gets one derived from its class name. Leading
underscores and a trailing `Config` are dropped, and the CamelCase name becomes
snake_case. A run of capitals stays together as one word, except for its last
capital when a lowercase letter follows; that capital starts the next word.

| Class | Tag |
| ----- | --- |
| `MCARConfig` | `mcar` |
| `CosineWarmupConfig` | `cosine_warmup` |
| `UKBFieldsConfig` | `ukb_fields` |
| `ICD10Config` | `icd10` |
| `Icd9ToIcd10Config` | `icd9_to_icd10` |
| `SGD` | `sgd` |
| `AdamW` | `adam_w` |
| `SomaScan` | `soma_scan` |

When the derived tag reads badly, set it explicitly. An explicit tag always
wins:

```python
from typing import ClassVar


class AdamW(ConfigModel):
    name: ClassVar[str] = "adamw"
    lr: float = 1e-3
```

An explicit tag counts only on the class that sets it, so a subclass of a
variant gets its own tag instead of reusing its parent's. `tagged_union` raises
when two variants end up with the same tag.

!!! warning "Renaming a class changes its derived tag"
    Saved configs refer to variants by tag. Renaming `MCARConfig` to
    `UniformMissingConfig` changes its tag from `mcar` to `uniform_missing`, and
    configs saved with `name: mcar` stop loading (with a clear error naming the
    valid tags). Where saved configs must keep loading, set the tag explicitly
    or pin the tags in a test:

    ```python
    def test_missingness_tags_are_stable():
        assert InformativeConfig.name == "informative"
        assert MCARConfig.name == "mcar"
    ```

## Declaring the tag as a field raises

Because the tag selects the variant, it can't also be a setting. A variant that
declares it as a field is rejected when the union is built:

```python
class MCARConfig(ConfigModel):
    name: str = "mcar"  # a field, not a ClassVar
    rate: float = 0.1

tagged_union(MCARConfig, InformativeConfig)
# TypeError: MCARConfig.name is a field; remove it (the tag is derived from the
# class name) or declare it as `name: ClassVar[str] = "..."`. The tag selects
# the variant and is not a setting.
```

A tag that is set but isn't a non-empty string, a variant that isn't a model or
dataclass, and fewer than two variants are rejected the same way.

## Switching variants

- **YAML:** set the tag, plus any fields of that variant.
- **CLI:** `missingness.name=informative`. A new tag starts that component from
  the new variant's defaults, and the other keys apply to the new variant, in
  any order: `missingness.name=mcar missingness.rate=0.3`. Re-sending the
  current tag keeps the component's values. Shortcuts work as for any key, so
  `name=mcar` works when only one union is in the config.
- **Env vars:** `MYAPP_MISSINGNESS__NAME=mcar MYAPP_MISSINGNESS__RATE=0.3`.
- **Python:** assign an instance, `cfg.missingness = MCARConfig(rate=0.3)`.

`--help` shows the tag at the top of the component's panel, with its choices:

```
╭─ missingness ─────────────────────────────────────────────────╮
│   name : Literal['informative', 'mcar'] = 'informative'        │
│       Selects the variant; switch with missingness.name=<tag>. │
│   severity_slope : float = 0.8                                 │
╰────────────────────────────────────────────────────────────────╯
```

A wrong tag fails with the valid choices, and a field from another variant is
reported under the selected one:

```
✗ Config Validation Error in 'SiteConfig' — 1 problem found
└── missingness  ✗ 'typo' is not a valid name; expected 'informative', 'mcar'
```

## Options

- **A different tag key:** `tagged_union(LocalConfig, S3Config, tag="backend")`.
  The YAML key and the class attribute both become `backend`.
- **Optional components:** `store: Optional[Store] = None`.
- **Dataclass and plain Pydantic variants** work too. Prefer `ConfigModel`
  variants: a plain `BaseModel` ignores unknown keys by default, so a typo in a
  variant's YAML would pass silently.
- **JSON schema:** `SiteConfig.model_json_schema()` describes the tag as a
  required `const` on each variant, with a discriminator mapping, so editors
  that validate YAML against the schema understand it.

## Type checking

A type checker can't follow a union built at runtime, so it sees a field
annotated with `tagged_union(...)` as `Any`. Give it a plain union instead:

```python
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    Missingness = Union[InformativeConfig, MCARConfig]
else:
    Missingness = tagged_union(InformativeConfig, MCARConfig)
```

A derived tag is set on the class at runtime, so the type checker doesn't know
about it. To read `.name` in type-checked code, declare
`name: ClassVar[str]` (without a value) on a shared base class, or use
`isinstance` to tell the variants apart.

> Runnable demo: [`examples/10_tagged_union.py`](https://github.com/bviggiano/onefig/blob/main/examples/10_tagged_union.py).
