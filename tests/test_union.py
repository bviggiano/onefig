from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Optional

import pytest
from pydantic import BaseModel, ValidationError

from onefig import ConfigModel, tagged_union
from onefig._union import derive_tag


class LocalConfig(ConfigModel):
    path: str = "./data"


class S3Config(ConfigModel):
    bucket: str  # required: a value that omits it fails to validate as s3


class GCSStore(ConfigModel):
    name: ClassVar[str] = "gcs"  # explicit tag wins over the derived "gcs_store"
    project: str = "lab"


Store = tagged_union(LocalConfig, S3Config, GCSStore)


class AppConfig(ConfigModel):
    store: Store = LocalConfig()


# ---- tag derivation --------------------------------------------------------


@pytest.mark.parametrize(
    ("class_name", "tag"),
    [
        ("MCARConfig", "mcar"),
        ("InformativeConfig", "informative"),
        ("CosineWarmupConfig", "cosine_warmup"),
        ("UKBFieldsConfig", "ukb_fields"),
        ("ICD10Config", "icd10"),
        ("Icd9ToIcd10Config", "icd9_to_icd10"),
        ("HTTPSProxy", "https_proxy"),
        ("SGD", "sgd"),
        ("AdamW", "adam_w"),
        ("Config", "config"),
        ("_PrivateConfig", "private"),
    ],
)
def test_derive_tag(class_name: str, tag: str) -> None:
    assert derive_tag(class_name) == tag


def test_tags_are_derived_unless_explicit() -> None:
    assert LocalConfig.name == "local"
    assert S3Config.name == "s3"
    assert GCSStore.name == "gcs"


def test_tag_is_not_a_field() -> None:
    assert list(LocalConfig.model_fields) == ["path"]
    assert list(GCSStore.model_fields) == ["project"]


def test_tag_reads_from_the_instance() -> None:
    assert AppConfig().store.name == "local"


def test_subclass_gets_its_own_tag() -> None:
    class Base(ConfigModel):
        name: ClassVar[str] = "base"

    class ChildConfig(Base):
        pass

    class OtherConfig(ConfigModel):
        pass

    tagged_union(Base, ChildConfig, OtherConfig)
    assert ChildConfig.name == "child"
    assert Base.name == "base"


# ---- selecting a variant ---------------------------------------------------


def test_selects_variant_by_tag(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("store:\n  name: s3\n  bucket: my-bucket\n")
    cfg = AppConfig.load(p)
    assert isinstance(cfg.store, S3Config)
    assert cfg.store.bucket == "my-bucket"


def test_selects_explicitly_tagged_variant() -> None:
    cfg = AppConfig.from_dict({"store": {"name": "gcs", "project": "p"}})
    assert isinstance(cfg.store, GCSStore) and cfg.store.project == "p"


def test_default_variant_is_kept() -> None:
    assert isinstance(AppConfig().store, LocalConfig)


def test_unknown_tag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"store": {"name": "azure"}})


def test_missing_tag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"store": {"path": "/tmp"}})


def test_selected_variant_is_validated() -> None:
    # name=s3 routes to S3Config, whose required ``bucket`` is missing here.
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"store": {"name": "s3"}})


def test_other_variants_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"store": {"name": "s3", "bucket": "b", "path": "x"}})


def test_tag_cannot_be_passed_to_the_constructor() -> None:
    with pytest.raises(ValidationError):
        LocalConfig(name="local")  # type: ignore[call-arg]


def test_assigning_a_variant_instance() -> None:
    cfg = AppConfig()
    cfg.store = S3Config(bucket="b")
    assert isinstance(cfg.store, S3Config)


def test_custom_tag_key() -> None:
    class FastConfig(ConfigModel):
        backend: ClassVar[str] = "fast"

    class SafeConfig(ConfigModel):
        retries: int = 3

    class Holder(ConfigModel):
        engine: tagged_union(FastConfig, SafeConfig, tag="backend") = FastConfig()

    cfg = Holder.from_dict({"engine": {"backend": "safe", "retries": 5}})
    assert isinstance(cfg.engine, SafeConfig) and cfg.engine.retries == 5
    assert SafeConfig.backend == "safe"  # type: ignore[attr-defined]
    assert cfg.to_dict() == {"engine": {"backend": "safe", "retries": 5}}


def test_optional_union() -> None:
    class Holder(ConfigModel):
        store: Optional[Store] = None  # noqa: UP045 (pydantic evaluates this on py3.9)

    assert Holder().store is None
    cfg = Holder.from_dict({"store": {"name": "s3", "bucket": "b"}})
    assert isinstance(cfg.store, S3Config)
    assert cfg.to_dict() == {"store": {"name": "s3", "bucket": "b"}}


# ---- serialization ---------------------------------------------------------


def test_dump_carries_the_tag() -> None:
    assert AppConfig().to_dict() == {"store": {"name": "local", "path": "./data"}}


def test_flat_dict_round_trip() -> None:
    cfg = AppConfig.from_dict({"store": {"name": "s3", "bucket": "b"}})
    flat = cfg.to_flat_dict()
    assert flat == {"store.name": "s3", "store.bucket": "b"}
    assert AppConfig.from_flat_dict(flat) == cfg


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = AppConfig.from_dict({"store": {"name": "gcs", "project": "p"}})
    p = tmp_path / "snap.yaml"
    cfg.save_yaml(p)
    assert "name: gcs" in p.read_text()
    assert AppConfig.load(p) == cfg


def test_diff_shows_the_tag_change() -> None:
    other = AppConfig.from_dict({"store": {"name": "gcs"}})
    diff = AppConfig().diff(other)
    assert diff["store.name"] == ("local", "gcs")


def test_freeze_reaches_the_variant() -> None:
    cfg = AppConfig().freeze()
    with pytest.raises(RuntimeError):
        cfg.store.path = "/elsewhere"  # type: ignore[union-attr]


def test_json_schema_has_discriminator() -> None:
    schema = AppConfig.model_json_schema()
    union = schema["properties"]["store"]
    assert union["discriminator"]["propertyName"] == "name"
    assert set(union["discriminator"]["mapping"]) == {"local", "s3", "gcs"}
    consts = {
        part["properties"]["name"]["const"]
        for choice in union["oneOf"]
        for part in choice["allOf"]
        if "properties" in part
    }
    assert consts == {"local", "s3", "gcs"}


# ---- variant kinds ---------------------------------------------------------


@dataclass
class DataclassVariant:
    value: int = 0


class PlainModelVariant(BaseModel):
    size: int = 1


def test_variants_may_be_dataclasses_and_plain_models() -> None:
    mixed = tagged_union(DataclassVariant, PlainModelVariant)

    class Holder(ConfigModel):
        item: mixed = PlainModelVariant()

    cfg = Holder.model_validate({"item": {"name": "dataclass_variant", "value": 7}})
    assert isinstance(cfg.item, DataclassVariant) and cfg.item.value == 7
    assert [f.name for f in fields(DataclassVariant)] == ["value"]
    assert cfg.to_dict() == {"item": {"name": "dataclass_variant", "value": 7}}


# ---- definition-time errors ------------------------------------------------


def test_requires_at_least_two_variants() -> None:
    with pytest.raises(ValueError, match="at least two variants"):
        tagged_union(LocalConfig)


def test_tag_as_a_model_field_raises() -> None:
    class LiteralTagged(ConfigModel):
        name: str = "literal"

    with pytest.raises(TypeError, match=r"LiteralTagged\.name is a field"):
        tagged_union(LiteralTagged, LocalConfig)


def test_tag_as_a_dataclass_field_raises() -> None:
    @dataclass
    class FieldTagged:
        name: str = "field"

    with pytest.raises(TypeError, match=r"FieldTagged\.name is a field"):
        tagged_union(FieldTagged, LocalConfig)


@pytest.mark.parametrize("bad", ["", 3])
def test_explicit_tag_must_be_a_non_empty_string(bad: object) -> None:
    class BadTag(ConfigModel):
        pass

    BadTag.name = bad  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="non-empty string"):
        tagged_union(BadTag, LocalConfig)


def test_duplicate_derived_tags_raise() -> None:
    def make() -> type[ConfigModel]:
        class MCARConfig(ConfigModel):
            pass

        return MCARConfig

    with pytest.raises(ValueError, match="'mcar'"):
        tagged_union(make(), make())


def test_explicit_tag_colliding_with_derived_raises() -> None:
    class Impostor(ConfigModel):
        name: ClassVar[str] = "local"

    with pytest.raises(ValueError, match="'local'"):
        tagged_union(LocalConfig, Impostor)


def test_non_model_variant_raises() -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="NotAModel"):
        tagged_union(NotAModel, LocalConfig)


def test_diff_from_defaults_knows_the_default_tag() -> None:
    cfg = AppConfig.from_dict({"store": {"name": "gcs"}})
    diff = cfg.diff_from_defaults()
    assert diff["store.name"] == ("local", "gcs")
    assert AppConfig().diff_from_defaults() == {}
