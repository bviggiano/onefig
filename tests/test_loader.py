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


# ---- extends: ---------------------------------------------------------------


def test_extends_single_parent(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "a: 1\nb: 2\n")
    child = _write(
        tmp_path / "child.yaml",
        "extends: base.yaml\nb: 99\nc: 3\n",
    )
    # Parent provides a:1; child overrides b and adds c. `extends` is consumed.
    assert _load(child) == {"a": 1, "b": 99, "c": 3}


def test_extends_deep_merge(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        "model:\n  name: base\n  lr: 0.01\n  arch:\n    depth: 4\n",
    )
    child = _write(
        tmp_path / "child.yaml",
        "extends: base.yaml\nmodel:\n  lr: 0.001\n  arch:\n    width: 256\n",
    )
    # `lr` overrides; `name` and `arch.depth` survive; `arch.width` is added.
    assert _load(child) == {
        "model": {
            "name": "base",
            "lr": 0.001,
            "arch": {"depth": 4, "width": 256},
        }
    }


def test_extends_list_left_to_right(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "x: 1\ny: 1\nz: 1\n")
    _write(tmp_path / "b.yaml", "y: 2\nz: 2\n")
    child = _write(
        tmp_path / "child.yaml",
        "extends:\n  - a.yaml\n  - b.yaml\nz: 99\n",
    )
    # b overrides a; child overrides both.
    assert _load(child) == {"x": 1, "y": 2, "z": 99}


def test_extends_chain(tmp_path: Path) -> None:
    _write(tmp_path / "grand.yaml", "a: 1\nb: 1\nc: 1\n")
    _write(tmp_path / "parent.yaml", "extends: grand.yaml\nb: 2\nc: 2\n")
    child = _write(
        tmp_path / "child.yaml",
        "extends: parent.yaml\nc: 3\n",
    )
    assert _load(child) == {"a": 1, "b": 2, "c": 3}


def test_extends_relative_to_child_dir(tmp_path: Path) -> None:
    _write(tmp_path / "shared" / "base.yaml", "a: 1\n")
    child = _write(
        tmp_path / "experiments" / "child.yaml",
        "extends: ../shared/base.yaml\nb: 2\n",
    )
    assert _load(child) == {"a": 1, "b": 2}


def test_extends_absolute_path(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", "a: 1\n")
    child = _write(
        tmp_path / "sub" / "child.yaml",
        f"extends: {base}\nb: 2\n",
    )
    assert _load(child) == {"a": 1, "b": 2}


def test_extends_missing_parent_raises(tmp_path: Path) -> None:
    child = _write(
        tmp_path / "child.yaml",
        "extends: nope.yaml\n",
    )
    with pytest.raises(FileNotFoundError, match="extends.*not found"):
        _load(child)


def test_extends_cycle_raises(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: b.yaml\nv: 1\n")
    b = _write(tmp_path / "b.yaml", "extends: a.yaml\nv: 2\n")
    with pytest.raises(ValueError, match="Cycle in extends chain"):
        _load(b)


def test_extends_self_cycle_raises(tmp_path: Path) -> None:
    p = _write(tmp_path / "self.yaml", "extends: self.yaml\nv: 1\n")
    with pytest.raises(ValueError, match="Cycle"):
        _load(p)


def test_extends_bad_type_raises(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "extends: 42\nv: 1\n")
    with pytest.raises(ValueError, match="must be a string or list"):
        _load(p)


def test_extends_interpolation_across_files(tmp_path: Path) -> None:
    # Interpolation is resolved after the chain is merged, so a parent can
    # reference a key that only the child provides.
    _write(tmp_path / "base.yaml", "greeting: hello_${name}\n")
    child = _write(
        tmp_path / "child.yaml",
        "extends: base.yaml\nname: world\n",
    )
    assert _load(child) == {"greeting": "hello_world", "name": "world"}


def test_extends_key_consumed(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "a: 1\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\nb: 2\n")
    out = _load(child)
    assert "extends" not in out
