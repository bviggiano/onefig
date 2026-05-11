from __future__ import annotations

import json
from pathlib import Path

import pytest

from onefig import ConfigModel


class Sub(ConfigModel):
    lr: float = 1e-4
    """Learning rate."""


class Cfg(ConfigModel):
    epochs: int = 10
    """Number of epochs."""
    model: Sub = Sub()


def test_export_writes_valid_json_schema(tmp_path: Path) -> None:
    out = tmp_path / "schemas" / "train.schema.json"
    Cfg.export_json_schema(out, register=False)

    assert out.exists()
    schema = json.loads(out.read_text())
    assert schema["type"] == "object"
    assert "epochs" in schema["properties"]
    # Field docstring flows through to JSON Schema as `description`.
    assert schema["properties"]["epochs"]["description"] == "Number of epochs."


def test_export_returns_resolved_path(tmp_path: Path) -> None:
    out = tmp_path / "train.schema.json"
    returned = Cfg.export_json_schema(out, register=False)
    assert returned == out


def test_register_creates_vscode_settings(tmp_path: Path) -> None:
    schema = tmp_path / "schemas" / "train.schema.json"
    settings = tmp_path / ".vscode" / "settings.json"
    Cfg.export_json_schema(
        schema,
        register=True,
        vscode_settings_path=settings,
    )

    data = json.loads(settings.read_text())
    yaml_schemas = data["yaml.schemas"]
    # Schema key is workspace-relative (parent of .vscode/ is workspace root).
    assert yaml_schemas == {
        "schemas/train.schema.json": ["**/train.yaml", "**/train.yml"],
    }


def test_register_strips_dot_schema_from_stem(tmp_path: Path) -> None:
    # `foo.schema.json` derives globs from `foo`, not `foo.schema`.
    schema = tmp_path / "foo.schema.json"
    settings = tmp_path / ".vscode" / "settings.json"
    Cfg.export_json_schema(schema, vscode_settings_path=settings)
    data = json.loads(settings.read_text())
    assert data["yaml.schemas"] == {
        "foo.schema.json": ["**/foo.yaml", "**/foo.yml"],
    }


def test_register_custom_glob_string(tmp_path: Path) -> None:
    schema = tmp_path / "train.schema.json"
    settings = tmp_path / ".vscode" / "settings.json"
    Cfg.export_json_schema(
        schema,
        yaml_glob="configs/*.yaml",
        vscode_settings_path=settings,
    )
    data = json.loads(settings.read_text())
    assert data["yaml.schemas"] == {"train.schema.json": ["configs/*.yaml"]}


def test_register_custom_glob_list(tmp_path: Path) -> None:
    schema = tmp_path / "train.schema.json"
    settings = tmp_path / ".vscode" / "settings.json"
    Cfg.export_json_schema(
        schema,
        yaml_glob=["a/*.yaml", "b/*.yml"],
        vscode_settings_path=settings,
    )
    data = json.loads(settings.read_text())
    assert data["yaml.schemas"] == {
        "train.schema.json": ["a/*.yaml", "b/*.yml"],
    }


def test_register_merges_into_existing_settings(tmp_path: Path) -> None:
    settings = tmp_path / ".vscode" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "editor.formatOnSave": True,
                "yaml.schemas": {"existing.json": ["other.yaml"]},
            }
        )
    )

    schema = tmp_path / "train.schema.json"
    Cfg.export_json_schema(schema, vscode_settings_path=settings)

    data = json.loads(settings.read_text())
    # Pre-existing keys are preserved.
    assert data["editor.formatOnSave"] is True
    # Pre-existing schema mappings are preserved alongside the new one.
    assert data["yaml.schemas"] == {
        "existing.json": ["other.yaml"],
        "train.schema.json": ["**/train.yaml", "**/train.yml"],
    }


def test_register_overwrites_same_schema_key(tmp_path: Path) -> None:
    settings = tmp_path / ".vscode" / "settings.json"
    schema = tmp_path / "train.schema.json"

    Cfg.export_json_schema(
        schema, yaml_glob="old.yaml", vscode_settings_path=settings
    )
    Cfg.export_json_schema(
        schema, yaml_glob="new.yaml", vscode_settings_path=settings
    )

    data = json.loads(settings.read_text())
    assert data["yaml.schemas"] == {"train.schema.json": ["new.yaml"]}


def test_register_false_skips_vscode(tmp_path: Path) -> None:
    schema = tmp_path / "train.schema.json"
    settings = tmp_path / ".vscode" / "settings.json"
    Cfg.export_json_schema(
        schema, register=False, vscode_settings_path=settings
    )
    assert schema.exists()
    assert not settings.exists()


def test_register_rejects_jsonc_settings(tmp_path: Path) -> None:
    settings = tmp_path / ".vscode" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{\n  // a comment\n  "editor.formatOnSave": true\n}')

    schema = tmp_path / "train.schema.json"
    with pytest.raises(ValueError, match="not valid JSON"):
        Cfg.export_json_schema(schema, vscode_settings_path=settings)


def test_schema_key_absolute_when_outside_workspace(tmp_path: Path) -> None:
    # Schema lives outside the workspace root (parent of .vscode/) — must
    # fall back to an absolute path rather than a `..`-prefixed relative.
    workspace = tmp_path / "workspace"
    settings = workspace / ".vscode" / "settings.json"
    schema = tmp_path / "external" / "train.schema.json"

    Cfg.export_json_schema(schema, vscode_settings_path=settings)

    data = json.loads(settings.read_text())
    (key,) = data["yaml.schemas"].keys()
    assert Path(key).is_absolute()
    assert Path(key) == schema.resolve()
