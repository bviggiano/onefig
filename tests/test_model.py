from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from onefig import ConfigModel
from onefig.model import FrozenConfigError


class Sub(ConfigModel):
    lr: float = 1e-4
    name: str = "default"


class Cfg(ConfigModel):
    epochs: int = 10
    model: Sub = Sub()


def test_load_validates(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 5\nmodel:\n  lr: 0.001\n  name: foo\n")
    cfg = Cfg.load(p)
    assert cfg.epochs == 5
    assert cfg.model.lr == 0.001
    assert cfg.model.name == "foo"


def test_load_type_error(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("epochs: not_an_int\n")
    with pytest.raises(ValidationError):
        Cfg.load(p)


def test_extra_forbid(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 5\nunknown_field: 1\n")
    with pytest.raises(ValidationError):
        Cfg.load(p)


def test_to_dict_and_flat_dict() -> None:
    cfg = Cfg(epochs=3, model=Sub(lr=0.01, name="bar"))
    assert cfg.to_dict() == {"epochs": 3, "model": {"lr": 0.01, "name": "bar"}}
    assert cfg.to_flat_dict() == {
        "epochs": 3,
        "model.lr": 0.01,
        "model.name": "bar",
    }


def test_freeze_blocks_mutation() -> None:
    cfg = Cfg()
    cfg.freeze()
    assert cfg.is_frozen
    with pytest.raises(FrozenConfigError):
        cfg.epochs = 99


def test_freeze_is_recursive() -> None:
    cfg = Cfg()
    cfg.freeze()
    with pytest.raises(FrozenConfigError):
        cfg.model.lr = 0.5


def test_assignment_revalidates() -> None:
    cfg = Cfg()
    with pytest.raises(ValidationError):
        cfg.epochs = "still not an int"  # type: ignore[assignment]


def test_from_dict_roundtrip() -> None:
    cfg = Cfg(epochs=7, model=Sub(lr=0.5, name="x"))
    rt = Cfg.from_dict(cfg.to_dict())
    assert rt == cfg


def test_format_tree() -> None:
    from onefig import format_tree

    cfg = Cfg(epochs=2, model=Sub(lr=0.1, name="z"))
    out = format_tree(cfg.model_dump(), name="MyCfg")
    assert "MyCfg:" in out
    assert "epochs" in out
    assert "lr" in out


def test_display_prints(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = Cfg(epochs=2, model=Sub(lr=0.1, name="z"))
    cfg.display(name="MyCfg")
    captured = capsys.readouterr()
    assert "MyCfg:" in captured.out
    assert "epochs" in captured.out


def test_config_name_defaults_to_none() -> None:
    cfg = Cfg()
    assert cfg.config_name is None


def test_config_name_set_from_filename(tmp_path: Path) -> None:
    p = tmp_path / "my_run.yaml"
    p.write_text("epochs: 5\n")
    cfg = Cfg.load(p)
    assert cfg.config_name == "my_run"


def test_config_name_override_via_load(tmp_path: Path) -> None:
    p = tmp_path / "my_run.yaml"
    p.write_text("epochs: 5\n")
    cfg = Cfg.load(p, config_name="custom_label")
    assert cfg.config_name == "custom_label"


def test_config_name_set_via_name_lookup(tmp_path: Path) -> None:
    nested = tmp_path / "configs"
    nested.mkdir()
    (nested / "baseline.yaml").write_text("epochs: 1\n")
    cfg = Cfg.load("baseline", search_root=tmp_path)
    assert cfg.config_name == "baseline"


def test_config_name_not_in_dump() -> None:
    cfg = Cfg()
    object.__setattr__(cfg, "_config_name", "anything")
    assert "config_name" not in cfg.to_dict()
    assert "config_name" not in cfg.to_flat_dict()


def test_config_name_renaming_when_unfrozen() -> None:
    cfg = Cfg()
    assert cfg.config_name is None
    cfg.config_name = "renamed"
    assert cfg.config_name == "renamed"


def test_config_name_renaming_blocked_when_frozen() -> None:
    cfg = Cfg()
    cfg.config_name = "before_freeze"
    cfg.freeze()
    with pytest.raises(FrozenConfigError):
        cfg.config_name = "after_freeze"
    assert cfg.config_name == "before_freeze"


def test_display_uses_config_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "experiment_42.yaml"
    p.write_text("epochs: 5\n")
    cfg = Cfg.load(p)
    cfg.display()
    assert "experiment_42:" in capsys.readouterr().out


def test_save_yaml_roundtrips(tmp_path: Path) -> None:
    cfg = Cfg(epochs=7, model=Sub(lr=0.5, name="bert"))
    out = tmp_path / "snapshot.yaml"
    cfg.save_yaml(out)
    assert out.is_file()

    rebuilt = Cfg.load(out)
    assert rebuilt == cfg
    assert rebuilt.config_name == "snapshot"


def test_save_yaml_accepts_string_path(tmp_path: Path) -> None:
    cfg = Cfg(epochs=2, model=Sub(lr=0.1, name="x"))
    out = tmp_path / "config.yaml"
    cfg.save_yaml(str(out))
    assert out.read_text().strip() != ""


def test_model_post_init_fires_through_load(tmp_path: Path) -> None:
    """Pydantic's model_post_init is the recommended derived-field hook."""

    class Derived(ConfigModel):
        x: int
        y: int
        total: int = 0

        def model_post_init(self, __context: object) -> None:
            object.__setattr__(self, "total", self.x + self.y)

    p = tmp_path / "d.yaml"
    p.write_text("x: 3\ny: 4\n")
    cfg = Derived.load(p)
    assert cfg.total == 7

    # Also fires via from_dict and direct construction
    assert Derived.from_dict({"x": 1, "y": 2}).total == 3
    assert Derived(x=10, y=20).total == 30
