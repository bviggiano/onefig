from __future__ import annotations

import json
from pathlib import Path


def export_schema(
    schema: dict,
    path: Path,
    *,
    register: bool,
    yaml_glob: str | list[str] | None,
    vscode_settings_path: Path,
) -> Path:
    """Write a JSON Schema to disk; optionally wire it into VSCode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n")
    if register:
        register_vscode_schema(
            schema_path=path,
            yaml_glob=yaml_glob,
            settings_path=vscode_settings_path,
        )
    return path


def register_vscode_schema(
    *,
    schema_path: Path,
    yaml_glob: str | list[str] | None,
    settings_path: Path,
) -> None:
    """Merge a ``yaml.schemas`` mapping into a VSCode ``settings.json``.

    The YAML extension treats the keys of ``yaml.schemas`` as schema
    locations and the values as one or more file globs. We resolve the
    schema location relative to the workspace root (the parent of the
    ``.vscode`` directory) when possible so the file is portable across
    machines, and fall back to an absolute path otherwise.
    """
    globs = _derive_globs(schema_path, yaml_glob)
    settings = _read_settings(settings_path)
    yaml_schemas = settings.setdefault("yaml.schemas", {})
    yaml_schemas[_schema_key(schema_path, settings_path)] = globs
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _derive_globs(
    schema_path: Path, yaml_glob: str | list[str] | None
) -> list[str]:
    if yaml_glob is None:
        stem = schema_path.stem
        if stem.endswith(".schema"):
            stem = stem[: -len(".schema")]
        return [f"**/{stem}.yaml", f"**/{stem}.yml"]
    if isinstance(yaml_glob, str):
        return [yaml_glob]
    return list(yaml_glob)


def _read_settings(settings_path: Path) -> dict:
    if not settings_path.exists():
        return {}
    text = settings_path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # VSCode tolerates comments in settings.json (JSONC); the stdlib
        # parser does not. Rather than silently stripping comments and
        # risking a mangled write-back, bail loudly.
        raise ValueError(
            f"Cannot register schema: {settings_path} is not valid JSON "
            f"(likely contains // comments). Edit it manually to add the "
            f"yaml.schemas mapping, or pass register=False."
        ) from exc


def _schema_key(schema_path: Path, settings_path: Path) -> str:
    # `.vscode/settings.json` treats schema paths as relative to the
    # workspace root (the parent of `.vscode`). Compute that when we can.
    workspace_root = settings_path.resolve().parent.parent
    schema_abs = schema_path.resolve()
    try:
        return str(schema_abs.relative_to(workspace_root))
    except ValueError:
        return str(schema_abs)
