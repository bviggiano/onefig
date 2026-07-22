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


_BANNER_OPEN = "# ========== onefig: tab completion =========="
_BANNER_CLOSE = "# ============================================"
_PY_BANNER_OPEN = "# ====== onefig: python tab completion ======="
_PY_BANNER_CLOSE = "# ============================================"


def _bash_script(prog: str) -> str:
    func = f"_onefig_complete_{_safe_func_suffix(prog)}"
    qprog = shlex.quote(prog)
    return f"""\
{_BANNER_OPEN}
{func}() {{
    local cur
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    local IFS=$'\\n'
    local candidates
    candidates=$({qprog} --onefig-completions "$cur" 2>/dev/null)
    COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )
}}
complete -o nospace -F {func} {qprog}
{_BANNER_CLOSE}
"""


def _zsh_script(prog: str) -> str:
    func = f"_onefig_complete_{_safe_func_suffix(prog)}"
    qprog = shlex.quote(prog)
    return f"""\
{_BANNER_OPEN}
{func}() {{
    local -a candidates
    candidates=("${{(@f)$({qprog} --onefig-completions \"$PREFIX\" 2>/dev/null)}}")
    [[ ${{#candidates[@]}} -gt 0 ]] && compadd -S '' -- $candidates
}}
if (( ! ${{+functions[compdef]}} )); then
    autoload -Uz compinit 2>/dev/null && compinit -u 2>/dev/null
fi
(( ${{+functions[compdef]}} )) && compdef {func} {prog}
{_BANNER_CLOSE}
"""


def _fish_script(prog: str) -> str:
    func = f"__onefig_complete_{_safe_func_suffix(prog)}"
    qprog = shlex.quote(prog)
    return f"""\
{_BANNER_OPEN}
function {func}
    set -l cur (commandline -t)
    {qprog} --onefig-completions "$cur" 2>/dev/null
end
complete -c {prog} -f -a '({func})'
{_BANNER_CLOSE}
"""


def python_completion_script(shell: str) -> str:
    """Render an install snippet for ``python <script>.py`` tab completion.

    Bound to ``python`` / ``python3`` (and friends), the generated function
    walks the current command line to find the script arg. Before
    invoking the script it greps for the literal word ``onefig`` in the
    file; if not found, the wrapper returns immediately without running
    the script. This keeps TAB safe for arbitrary Python scripts that
    have side effects at import time. Onefig scripts (anything that
    ``import``s or references ``onefig``) match the grep and are invoked
    with ``--onefig-completions <prefix>``; the printed candidates are
    used as the completion list.

    Falls back to file completion while the user is still typing the
    script path.

    One-time install. Sourcing the snippet enables completion for every
    onefig-based script invoked via ``python``, regardless of whether the
    script is on ``$PATH`` or directly executable.

    Args:
        shell: One of ``"bash"``, ``"zsh"``, ``"fish"``.

    Returns:
        A shell snippet ready to write to a file or eval.

    Raises:
        ValueError: If ``shell`` isn't one of the supported shells.
    """
    if shell == "bash":
        return _python_bash_script()
    if shell == "zsh":
        return _python_zsh_script()
    if shell == "fish":
        return _python_fish_script()
    raise ValueError(f"Unsupported shell {shell!r}; expected one of: bash, zsh, fish.")


_PYTHON_NAMES = (
    "python",
    "python3",
    "python3.9",
    "python3.10",
    "python3.11",
    "python3.12",
    "python3.13",
)


def _python_bash_script() -> str:
    targets = " ".join(_PYTHON_NAMES)
    return f"""\
{_PY_BANNER_OPEN}
_onefig_python_complete() {{
    local cur script i
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    script=""
    for ((i=1; i<COMP_CWORD; i++)); do
        case "${{COMP_WORDS[i]}}" in
            -*) ;;
            *.py)
                if [[ -f "${{COMP_WORDS[i]}}" ]]; then
                    script="${{COMP_WORDS[i]}}"
                    break
                fi
                ;;
        esac
    done
    if [[ -z "$script" ]]; then
        COMPREPLY=( $(compgen -f -- "$cur") )
        return
    fi
    # Skip non-onefig scripts to avoid running their side effects on TAB.
    grep -qw onefig "$script" 2>/dev/null || return
    local IFS=$'\\n'
    local candidates
    candidates=$("${{COMP_WORDS[0]}}" "$script" --onefig-completions "$cur" 2>/dev/null)
    if [[ -z "$candidates" ]]; then
        return
    fi
    COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )
}}
complete -o nospace -F _onefig_python_complete {targets}
{_PY_BANNER_CLOSE}
"""


def _python_zsh_script() -> str:
    targets = " ".join(_PYTHON_NAMES)
    cand_expr = (
        '("${(@f)$($python_cmd "$script" --onefig-completions "$PREFIX" 2>/dev/null)}")'
    )
    return f"""\
{_PY_BANNER_OPEN}
_onefig_python_complete() {{
    local script python_cmd i
    python_cmd="${{words[1]}}"
    script=""
    for ((i=2; i<CURRENT; i++)); do
        case "${{words[i]}}" in
            -*) ;;
            *.py)
                if [[ -f "${{words[i]}}" ]]; then
                    script="${{words[i]}}"
                    break
                fi
                ;;
        esac
    done
    if [[ -z "$script" ]]; then
        _files
        return
    fi
    # Skip non-onefig scripts to avoid running their side effects on TAB.
    grep -qw onefig "$script" 2>/dev/null || return
    local -a candidates
    candidates={cand_expr}
    [[ ${{#candidates[@]}} -gt 0 ]] && compadd -S '' -- $candidates
}}
if (( ! ${{+functions[compdef]}} )); then
    autoload -Uz compinit 2>/dev/null && compinit -u 2>/dev/null
fi
(( ${{+functions[compdef]}} )) && compdef _onefig_python_complete {targets}
{_PY_BANNER_CLOSE}
"""


def _python_fish_script() -> str:
    cmds = " ".join(f"-c {n}" for n in _PYTHON_NAMES)
    return f"""\
{_PY_BANNER_OPEN}
function __onefig_python_complete
    set -l tokens (commandline -opc)
    set -l cur (commandline -t)
    set -l script ""
    for tok in $tokens[2..]
        if string match -q -- '-*' $tok
            continue
        end
        if string match -q -- '*.py' $tok; and test -f "$tok"
            set script $tok
            break
        end
    end
    if test -z "$script"
        __fish_complete_path "$cur"
        return
    end
    # Skip non-onefig scripts to avoid running their side effects on TAB.
    if not grep -qw onefig "$script" 2>/dev/null
        return
    end
    $tokens[1] "$script" --onefig-completions "$cur" 2>/dev/null
end
complete {cmds} -f -a '(__onefig_python_complete)'
{_PY_BANNER_CLOSE}
"""
