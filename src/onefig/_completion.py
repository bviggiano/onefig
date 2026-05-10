from __future__ import annotations

import shlex

from pydantic import BaseModel

from onefig._overrides import _resolve_keys

_SPECIAL_FLAGS = ("--show", "--help", "-h")


def completion_candidates(model: BaseModel) -> list[str]:
    """Return every override-key candidate the shell should offer.

    Each scalar field contributes its full dotted path suffixed with ``=``
    (e.g. ``"optimizer.lr="``). When the leaf name resolves uniquely it is
    also offered as a shorthand (``"lr="``); ambiguous leaves are omitted
    so users aren't shown a shortcut the override engine would refuse.

    Special flags (``--show``, ``--help``, ``-h``) round out the list.

    Args:
        model: A Pydantic model instance to introspect.

    Returns:
        A flat list of completion strings, in stable insertion order.
    """
    candidates = _resolve_keys(model)
    seen: set[str] = set()
    out: list[str] = []
    for key, paths in candidates.items():
        if len(paths) != 1:
            # Ambiguous leaf — skip; the full-path form will appear separately.
            continue
        token = f"{key}="
        if token not in seen:
            seen.add(token)
            out.append(token)
    out.extend(_SPECIAL_FLAGS)
    return out


def shell_script(shell: str, *, prog: str) -> str:
    """Render a shell-completion install snippet for ``prog``.

    The generated script binds tab-completion of ``prog`` to a callback
    that invokes ``prog --onefig-completions <prefix>`` and uses the
    output as the candidate list. Source the snippet in your shell rc to
    install (or eval it inline for one-shot use).

    Args:
        shell: One of ``"bash"``, ``"zsh"``, ``"fish"``.
        prog: The command name the user types to invoke the script (used
            both as the completion target and as the callback command).

    Returns:
        A shell snippet ready to write to a file or eval.

    Raises:
        ValueError: If ``shell`` isn't one of the supported shells.
    """
    if shell == "bash":
        return _bash_script(prog)
    if shell == "zsh":
        return _zsh_script(prog)
    if shell == "fish":
        return _fish_script(prog)
    raise ValueError(f"Unsupported shell {shell!r}; expected one of: bash, zsh, fish.")


def _safe_func_suffix(prog: str) -> str:
    """Turn ``prog`` into an identifier-safe suffix for the completion function."""
    out = []
    for ch in prog:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "onefig"


def _bash_script(prog: str) -> str:
    func = f"_onefig_complete_{_safe_func_suffix(prog)}"
    qprog = shlex.quote(prog)
    return f"""\
{func}() {{
    local cur
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    local IFS=$'\\n'
    local candidates
    candidates=$({qprog} --onefig-completions "$cur" 2>/dev/null)
    COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )
}}
complete -o nospace -F {func} {qprog}
"""


def _zsh_script(prog: str) -> str:
    func = f"_onefig_complete_{_safe_func_suffix(prog)}"
    qprog = shlex.quote(prog)
    return f"""\
#compdef {prog}
{func}() {{
    local -a candidates
    candidates=("${{(@f)$({qprog} --onefig-completions \"$PREFIX\" 2>/dev/null)}}")
    compadd -S '' -- $candidates
}}
{func} "$@"
"""


def _fish_script(prog: str) -> str:
    func = f"__onefig_complete_{_safe_func_suffix(prog)}"
    qprog = shlex.quote(prog)
    return f"""\
function {func}
    set -l cur (commandline -t)
    {qprog} --onefig-completions "$cur" 2>/dev/null
end
complete -c {prog} -f -a '({func})'
"""
