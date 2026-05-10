from __future__ import annotations

import argparse
import sys

from onefig._completion import python_completion_script, shell_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onefig",
        description=(
            "onefig command-line utilities. The library's primary API is the "
            "Python ConfigModel; this CLI exposes a small number of install "
            "helpers for shell tab completion."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_install = sub.add_parser(
        "install-completion",
        help=(
            "Print a shell-completion install snippet bound to a specific "
            "script command name."
        ),
        description=(
            "Print a shell-completion install snippet for a script. Source "
            "the output (or append it to your shell rc file) to enable TAB "
            "completion for the named command."
        ),
    )
    p_install.add_argument(
        "shell",
        choices=("bash", "zsh", "fish"),
        help="Target shell.",
    )
    p_install.add_argument(
        "--prog",
        required=True,
        metavar="NAME",
        help="Command name the user types to invoke the script.",
    )

    p_py = sub.add_parser(
        "install-python-completion",
        help=(
            "Print a generic shell snippet that enables tab completion for "
            "every onefig script invoked via `python <script>.py`."
        ),
        description=(
            "Print a generic shell snippet that hooks tab completion onto "
            "`python` and `python3`. After installing it once, every "
            "onefig-based script invoked via `python script.py` gets "
            "completion automatically."
        ),
    )
    p_py.add_argument(
        "shell",
        choices=("bash", "zsh", "fish"),
        help="Target shell.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``onefig`` console script.

    Args:
        argv: CLI arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success).
    """
    args = build_parser().parse_args(argv)
    if args.command == "install-completion":
        print(shell_script(args.shell, prog=args.prog))
    elif args.command == "install-python-completion":
        print(python_completion_script(args.shell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
