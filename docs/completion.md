# Shell tab completion

Installing onefig adds an `onefig` console command with a one-time
install helper for shell tab completion. Follow the steps below to set
it up.

**1. Install onefig.**

```bash
pip install onefig
```

This registers the `onefig` console command on your `$PATH`.

**2. Append the snippet for your shell to its rc file.** This binds
completion once to `python` itself; every onefig-based script invoked
via `python script.py` then gets TAB completion automatically. Run
`echo $SHELL` if you're not sure which shell you're on.

```bash
# bash (Linux default, older macOS)
onefig install-python-completion bash >> ~/.bashrc
source ~/.bashrc
```

```bash
# zsh (macOS default since Catalina)
onefig install-python-completion zsh >> ~/.zshrc
source ~/.zshrc
```

```bash
# fish
onefig install-python-completion fish >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

> The shell arg passed to `onefig` *must* match the rc file you append
> to. The bash snippet uses bash builtins like `complete -F`, which
> aren't available in zsh; the zsh snippet uses `compdef` / `compadd`.

**3. Try it.**

```bash
python examples/03_completion.py opt<TAB>   # → optimizer.kind=  optimizer.lr=
python examples/03_completion.py l<TAB>     # → lr=
python examples/03_completion.py --<TAB>    # → --show  --help
```

> Runnable demo: [`examples/03_completion.py`](https://github.com/bviggiano/onefig/blob/main/examples/03_completion.py).

**Preview before sourcing.** Both subcommands print their snippet to
stdout, so you can inspect what would be appended to your rc file, or
eval it for a one-shot test in the current shell only:

```bash
onefig install-python-completion bash           # print only
eval "$(onefig install-python-completion bash)" # one-shot in current shell
```

**How it works.** The completion list contains every overridable full
dotted path (suffixed with `=`), every unambiguous leaf-name and
suffix-subpath shortcut, and the special flags `--show`, `--help`, `-h`.
Ambiguous shortcuts are deliberately omitted so users aren't offered a
form the override engine would refuse. The `python`-bound wrapper finds the first `.py`
argument on the command line, and before invoking the script it greps
the file for the literal word `onefig`. If the script doesn't reference
onefig, the wrapper returns silently — your TAB key is never going to
execute a random Python script with side effects at import time. For
onefig scripts, the wrapper invokes
`python <that script> --onefig-completions <prefix>` and uses the
output as the candidate list; the `--onefig-completions` flag is
intercepted inside `update_from_cli()` before `main()` runs, so even
the onefig script itself doesn't execute any of its own logic.

> If your script gets its `ConfigModel` through a re-export (e.g.
> `from mypkg.configs import TrainCfg`) and never mentions `onefig`
> directly, the grep gate will skip it. Add a `# onefig` comment
> anywhere in the file to opt in.

The same install snippet is also available as a flag on any onefig
script (``--onefig-install-python-completion <shell>``), which is
useful when the ``onefig`` command isn't on the user's ``$PATH`` (e.g.
inside an application virtualenv).
