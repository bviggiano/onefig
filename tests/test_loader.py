from __future__ import annotations

from pathlib import Path

import pytest

from onefig._loader import load_yaml, resolve_path


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _load(name_or_path: str | Path, search_root: Path | None = None) -> dict:
    return load_yaml(resolve_path(name_or_path, search_root=search_root))


def test_load_basic(tmp_path: Path) -> None:
    p = _write(tmp_path / "c.yaml", "a: 1\nb:\n  c: 2\n")
    assert _load(p) == {"a": 1, "b": {"c": 2}}


def test_load_resolves_interpolation(tmp_path: Path) -> None:
    p = _write(tmp_path / "c.yaml", "base: hello\nderived: ${base}_world\n")
    assert _load(p) == {"base": "hello", "derived": "hello_world"}


def test_load_by_name(tmp_path: Path) -> None:
    _write(tmp_path / "deep" / "nested" / "myconf.yaml", "x: 5\n")
    assert _load("myconf", search_root=tmp_path) == {"x": 5}


def test_load_by_name_with_extension(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "myconf.yml", "x: 7\n")
    assert _load("myconf.yml", search_root=tmp_path) == {"x": 7}


def test_missing_name_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_path("nope", search_root=tmp_path)


def test_ambiguous_name_raises(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "x.yaml", "v: 1\n")
    _write(tmp_path / "b" / "x.yaml", "v: 2\n")
    with pytest.raises(ValueError, match="Multiple"):
        resolve_path("x", search_root=tmp_path)
