from __future__ import annotations

import pytest

from onefig._cli_entry import main


def test_install_completion_prints_bash_snippet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["install-completion", "bash", "--prog", "train.py"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "_onefig_complete_train_py" in out
    assert "complete -o nospace -F" in out
    assert "train.py" in out


def test_install_completion_supports_zsh(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["install-completion", "zsh", "--prog", "mytool"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#compdef mytool" in out


def test_install_completion_supports_fish(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["install-completion", "fish", "--prog", "mytool"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "complete -c mytool" in out


def test_install_completion_requires_prog() -> None:
    with pytest.raises(SystemExit):
        main(["install-completion", "bash"])


def test_install_completion_rejects_unknown_shell() -> None:
    with pytest.raises(SystemExit):
        main(["install-completion", "tcsh", "--prog", "x"])


def test_install_python_completion_prints_wrapper(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["install-python-completion", "bash"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "_onefig_python_complete" in out
    assert "complete -o nospace -F _onefig_python_complete python python3" in out


def test_install_python_completion_supports_all_shells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for shell in ("bash", "zsh", "fish"):
        capsys.readouterr()  # drain
        rc = main(["install-python-completion", shell])
        assert rc == 0
        out = capsys.readouterr().out
        assert "_onefig_python_complete" in out or "__onefig_python_complete" in out


def test_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])
