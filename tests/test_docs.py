from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional

import pytest

from onefig import ConfigModel, Field, FieldTrait, describe_fields, field_docs


@dataclasses.dataclass(frozen=True)
class Drawn(FieldTrait):
    def describe(self) -> str:
        return "Drawn when unset."

    def format_value(self, value: object) -> str | None:
        return "<drawn>" if value is None else None

    def format_type(self, annotation: object) -> str:
        return "int"


class Run(ConfigModel):
    name: str
    """The run's name."""
    size: Optional[int] = Field(None, Drawn())  # noqa: UP045
    """People per cohort."""
    rate: float = Field(0.1, description="Set by keyword.")
    quiet: bool = False


@dataclasses.dataclass(frozen=True)
class Table:
    """A class docstring, which is no field's."""

    symbol: Any
    """The ``HGNC`` symbol | a real name."""
    loeuf: Any
    """Observed over expected,
    lower is more constrained."""
    bare: Any


@dataclasses.dataclass(frozen=True)
class Retable(Table):
    symbol: Any
    """Replaced in the subclass."""


class Column:
    """Stands in for an array: only its dtype is read."""

    def __init__(self, kind: str, name: str) -> None:
        self.dtype = type("dtype", (), {"kind": kind, "name": name})()


def cells(line: str) -> list[str]:
    """The cells of a Markdown table row, splitting on unescaped pipes only."""
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]


# ---- field_docs -------------------------------------------------------------------


def test_field_docs_reads_config_docstrings_and_descriptions() -> None:
    assert field_docs(Run) == {
        "name": "The run's name.",
        "size": "People per cohort.",
        "rate": "Set by keyword.",
    }


def test_field_docs_reads_dataclass_docstrings_in_field_order() -> None:
    assert field_docs(Table) == {
        "symbol": "The ``HGNC`` symbol | a real name.",
        "loeuf": "Observed over expected,\nlower is more constrained.",
    }


def test_a_subclass_docstring_replaces_the_base_one() -> None:
    docs = field_docs(Retable)
    assert docs["symbol"] == "Replaced in the subclass."
    assert docs["loeuf"].startswith("Observed over expected")


def test_field_docs_rejects_other_classes() -> None:
    with pytest.raises(TypeError, match="pydantic model or a dataclass"):
        field_docs(dict)


# ---- describe_fields ----------------------------------------------------------------


def test_a_config_table_has_defaults_and_uses_its_traits() -> None:
    rows = {row[0]: row for row in describe_fields(Run).rows}
    assert rows["name"][2] == "required"
    assert rows["size"] == (
        "size",
        "int",
        "<drawn>",
        "People per cohort. Drawn when unset.",
    )
    assert rows["rate"][1:3] == ("float", "0.1")
    assert describe_fields(Run).header == ("Field", "Type", "Default", "Description")


def test_a_dataclass_instance_is_typed_by_its_arrays_dtypes() -> None:
    table = Table(symbol=Column("U", "<U10"), loeuf=Column("f", "float64"), bare=3)
    kinds = {row[0]: row[1] for row in describe_fields(table).rows}
    assert kinds == {"symbol": "str", "loeuf": "float64", "bare": "Any"}


def test_every_row_keeps_its_cells_despite_pipes_in_types_and_text() -> None:
    for table in (describe_fields(Run), describe_fields(Table)):
        rows = [line for line in str(table).splitlines() if not line.startswith("|-")]
        assert len(rows) == len(table.rows) + 1
        assert all(len(cells(line)) == len(table.header) for line in rows)


def test_code_and_prose_are_escaped_for_a_table_cell() -> None:
    class Typed(ConfigModel):
        value: Optional[int] = None  # noqa: UP045
        """A ``a | b`` span, then a | b in prose."""

    row = str(describe_fields(Typed)).splitlines()[2]
    assert "<code>int &#124; None</code>" in row
    assert "<code>a &#124; b</code>" in row
    assert "then a \\| b in prose." in row


def test_a_table_renders_as_markdown_in_a_notebook() -> None:
    table = describe_fields(Table)
    assert table._repr_markdown_() == str(table)
    assert str(table).splitlines()[0] == "| Field | Type | Description |"
