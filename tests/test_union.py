from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from onefig import ConfigModel, tagged_union


class Local(BaseModel):
    backend: Literal["local"] = "local"
    path: str = "./data"


class S3(BaseModel):
    backend: Literal["s3"] = "s3"
    bucket: str  # required — a value that omits it fails to validate as S3


Store = tagged_union(Local, S3, tag="backend")


class AppConfig(ConfigModel):
    store: Store = Local()


def test_selects_variant_by_tag(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("store:\n  backend: s3\n  bucket: my-bucket\n")
    cfg = AppConfig.load(p)
    assert isinstance(cfg.store, S3)
    assert cfg.store.bucket == "my-bucket"


def test_default_variant_is_kept() -> None:
    assert isinstance(AppConfig().store, Local)


def test_unknown_tag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"store": {"backend": "gcs"}})


def test_selected_variant_is_validated() -> None:
    # backend=s3 routes to S3, whose required ``bucket`` is missing here.
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"store": {"backend": "s3"}})


def test_custom_tag_name() -> None:
    cfg = AppConfig.model_validate({"store": {"backend": "local", "path": "/tmp"}})
    assert isinstance(cfg.store, Local) and cfg.store.path == "/tmp"


def test_requires_at_least_two_variants() -> None:
    with pytest.raises(ValueError, match="at least two variants"):
        tagged_union(Local)


@dataclass
class DataclassVariant:
    kind: Literal["dc"] = "dc"
    value: int = 0


class OtherVariant(BaseModel):
    kind: Literal["other"] = "other"


def test_variants_may_be_stdlib_dataclasses() -> None:
    # Pydantic validates dataclasses too, so a schema defined elsewhere as
    # plain dataclasses composes here without redefining it.
    mixed = tagged_union(DataclassVariant, OtherVariant, tag="kind")

    class Holder(ConfigModel):
        item: mixed = OtherVariant()

    cfg = Holder.model_validate({"item": {"kind": "dc", "value": 7}})
    assert isinstance(cfg.item, DataclassVariant) and cfg.item.value == 7
