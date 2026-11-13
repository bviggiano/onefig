# Documenting fields

Every field can say what it holds: a config field through its docstring (or
`Field(description=...)`), and a dataclass field through the string literal just
below it.

```python
@dataclasses.dataclass(frozen=True)
class Genes:
    symbol: np.ndarray
    """The gene's HGNC symbol, such as ``ALB``."""
    loeuf: np.ndarray
    """gnomAD's LOEUF; lower is more constrained, NaN where unknown."""
```

## Read the docstrings

`onefig.field_docs(cls)` returns `{field: docstring}` in field order, for a pydantic
model (a `ConfigModel` included) or a dataclass. A dataclass subclass's docstring for
an inherited field replaces the base's; fields without one are left out.

```python
field_docs(Genes)
# {'symbol': "The gene's HGNC symbol, such as ``ALB``.", 'loeuf': "gnomAD's LOEUF; ..."}
```

## Show them as a table

`onefig.describe_fields(obj)` takes a config or dataclass, or an instance of one,
and returns a table of its fields that renders as Markdown in a notebook:

```python
describe_fields(genes)       # Field | Type | Description
describe_fields(WorldConfig) # Field | Type | Default | Description
```

- A config's table has each field's default, `required` for a required field, and
  uses its [field traits](traits.md) for the type, the default and an extra note,
  as `--help` does.
- For a dataclass instance, a field holding an array is typed by its dtype (`str`,
  `int32`, `float64`).
- `str(table)` is the Markdown text, and `table.rows` holds the cells unescaped.

Cells are escaped for Markdown tables: a pipe in prose is backslash-escaped, and
code (field names, types, defaults, and double-backtick spans in a docstring) is
written as HTML `<code>` with `|`, `&`, `<` and `>` encoded. Markdown code spans
ignore backslash escapes, so a type such as `int | None` in backticks would
otherwise split its cell.
