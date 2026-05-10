from __future__ import annotations

from typing import Literal

import pytest

from onefig import ConfigModel
from onefig._completion import (
    completion_candidates,
    python_completion_script,
    shell_script,
)


class _Sched(ConfigModel):
    name: Literal["cosine", "linear"] = "cosine"
    warmup: int = 100


class _Opt(ConfigModel):
    kind: Literal["sgd", "adam"] = "sgd"
    lr: float = 1e-4
    sched: _Sched = _Sched()


class _Cfg(ConfigModel):
    epochs: int = 10
    name: str = "default"  # makes 'name' ambiguous with sched.name
    opt: _Opt = _Opt()


def test_full_paths_are_offered_with_equals_suffix() -> None:
    cands = completion_candidates(_Cfg())
    assert "epochs=" in cands
    assert "opt.kind=" in cands
    assert "opt.lr=" in cands
    assert "opt.sched.name=" in cands
    assert "opt.sched.warmup=" in cands


def test_unambiguous_leaf_shorthands_are_offered() -> None:
    cands = completion_candidates(_Cfg())
    assert "kind=" in cands  # only path is opt.kind
    assert "lr=" in cands  # only path is opt.lr
    assert "warmup=" in cands  # only path is opt.sched.warmup


def test_ambiguous_leaves_are_omitted_from_shortcuts() -> None:
    cands = completion_candidates(_Cfg())
    # 'name' is both top-level (name=) and nested (opt.sched.name=). The
    # top-level full path 'name=' appears, but no extra shorthand entry.
    name_entries = [c for c in cands if c == "name="]
    assert len(name_entries) == 1, name_entries


def test_special_flags_appended() -> None:
    cands = completion_candidates(_Cfg())
    assert cands[-3:] == ["--show", "--help", "-h"]


def test_completions_flag_filters_by_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["--onefig-completions", "opt."], exit_on_completion=False)
    out = capsys.readouterr().out.splitlines()
    assert "opt.kind=" in out
    assert "opt.lr=" in out
    assert "opt.sched.name=" in out
    assert "epochs=" not in out  # filtered out by prefix


def test_completions_flag_with_no_prefix_emits_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(["--onefig-completions"], exit_on_completion=False)
    out = capsys.readouterr().out.splitlines()
    assert "epochs=" in out
    assert "--show" in out


def test_completions_flag_exits_when_requested() -> None:
    cfg = _Cfg()
    with pytest.raises(SystemExit) as exc:
        cfg.update_from_cli(["--onefig-completions"])
    assert exc.value.code == 0


def test_completions_flag_short_circuits_overrides() -> None:
    cfg = _Cfg()
    # 'nope' would normally raise; --onefig-completions short-circuits.
    cfg.update_from_cli(
        ["nope=1", "--onefig-completions", "epoch"], exit_on_completion=False
    )
    assert cfg.epochs == 10  # overrides not applied


def test_install_completion_emits_bash_script(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(
        ["--onefig-install-completion", "bash"], exit_on_completion=False
    )
    out = capsys.readouterr().out
    assert "_onefig_complete_" in out
    assert "complete -o nospace -F" in out
    assert "--onefig-completions" in out


def test_install_completion_emits_zsh_script(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(
        ["--onefig-install-completion", "zsh"], exit_on_completion=False
    )
    out = capsys.readouterr().out
    assert "#compdef" in out
    assert "compadd" in out


def test_install_completion_emits_fish_script(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(
        ["--onefig-install-completion", "fish"], exit_on_completion=False
    )
    out = capsys.readouterr().out
    assert "function __onefig_complete_" in out
    assert "complete -c " in out


def test_install_completion_unknown_shell_raises() -> None:
    cfg = _Cfg()
    with pytest.raises(ValueError, match="Unsupported shell"):
        cfg.update_from_cli(
            ["--onefig-install-completion", "tcsh"], exit_on_completion=False
        )


def test_shell_script_quotes_prog_with_special_chars() -> None:
    # Shell-special chars in prog should be safely quoted.
    out = shell_script("bash", prog="my script.py")
    assert "'my script.py'" in out
    # Function name uses identifier-safe suffix.
    assert "_onefig_complete_my_script_py" in out


def test_shell_completion_script_method_uses_argv0_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["/usr/local/bin/train.py"])
    cfg = _Cfg()
    out = cfg.shell_completion_script("bash")
    # Default uses basename of sys.argv[0].
    assert "train.py" in out


def test_shell_completion_script_method_accepts_explicit_prog() -> None:
    cfg = _Cfg()
    out = cfg.shell_completion_script("bash", prog="my-cmd")
    assert "my-cmd" in out


def test_python_wrapper_bash_targets_python_commands() -> None:
    out = python_completion_script("bash")
    assert "_onefig_python_complete" in out
    assert "complete -o nospace -F _onefig_python_complete python python3" in out
    # Walks the command line, calls back into the script with --onefig-completions.
    assert "--onefig-completions" in out
    # Falls back to file completion while the user is still typing the script.
    assert "compgen -f" in out


def test_python_wrapper_zsh_uses_compdef() -> None:
    out = python_completion_script("zsh")
    assert "#compdef python python3" in out
    assert "--onefig-completions" in out
    assert "_files" in out  # fallback while typing the script


def test_python_wrapper_fish_targets_python_commands() -> None:
    out = python_completion_script("fish")
    assert "function __onefig_python_complete" in out
    assert "-c python" in out
    assert "-c python3" in out
    assert "--onefig-completions" in out


def test_python_wrapper_unknown_shell_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported shell"):
        python_completion_script("tcsh")


def test_install_python_completion_via_cli_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _Cfg()
    cfg.update_from_cli(
        ["--onefig-install-python-completion", "bash"], exit_on_completion=False
    )
    out = capsys.readouterr().out
    assert "_onefig_python_complete" in out


def test_python_wrapper_completion_script_method() -> None:
    cfg = _Cfg()
    out = cfg.python_wrapper_completion_script("bash")
    assert "_onefig_python_complete" in out
