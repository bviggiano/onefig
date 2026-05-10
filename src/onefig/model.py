from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from typing_extensions import Self

from onefig._cli import parse_overrides
from onefig._format import flatten, format_tree, unflatten
from onefig._git import get_commit_hash
from onefig._help import format_help
from onefig._loader import load_yaml, resolve_path
from onefig._overrides import _resolve_keys, _set_dotted, apply_overrides


class FrozenConfigError(RuntimeError):
    """Raised when an attempt is made to mutate a frozen config."""


class ConfigModel(BaseModel):
    """Base class for typed onefig configurations.

    Backed by Pydantic v2 with ``validate_assignment=True`` and
    ``extra="forbid"``. Adds YAML loading, CLI overrides, flattening,
    freezing, and tree display on top of standard Pydantic behavior.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    _frozen: bool = PrivateAttr(default=False)
    _config_name: str | None = PrivateAttr(default=None)
    _commit_hash: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _capture_commit_hash(self) -> Self:
        # Runs after every construction path (direct __init__, model_validate,
        # load, from_dict, ...). Using a model_validator instead of
        # model_post_init means subclasses can't silently bypass this by
        # overriding model_post_init without super().
        if self._commit_hash is None:
            object.__setattr__(self, "_commit_hash", get_commit_hash())
        return self

    @property
    def commit_hash(self) -> str | None:
        """Git ``HEAD`` hash captured when this config was constructed.

        Best-effort: ``None`` if ``git`` is unavailable, the working
        directory isn't a repo, or capture otherwise failed.

        Returns:
            The 40-character commit hash, or ``None``.
        """
        return self._commit_hash

    @property
    def config_name(self) -> str | None:
        """Identifier for this config.

        Auto-set to the YAML file stem on :meth:`load`. Settable while the
        config is unfrozen; assignment to a frozen config raises
        :class:`FrozenConfigError`.

        Returns:
            The current name, or ``None`` if not set.
        """
        return self._config_name

    @config_name.setter
    def config_name(self, value: str | None) -> None:
        # Freeze enforcement happens in `__setattr__`; reaching the setter
        # means the config is mutable.
        object.__setattr__(self, "_config_name", value)

    @classmethod
    def load(
        cls,
        name_or_path: str | Path,
        *,
        search_root: str | Path | None = None,
        config_name: str | None = None,
    ) -> Self:
        """Load and validate a config from a YAML file.

        Args:
            name_or_path: Either a path to a YAML file (absolute or existing
                relative path) or a bare config name to search for under
                ``search_root``.
            search_root: Directory under which to recursively search when
                ``name_or_path`` is a bare name. Defaults to the current
                working directory.
            config_name: Optional override for ``cfg.config_name``. Defaults
                to the loaded file's stem.

        Returns:
            A validated instance of ``cls``.

        Raises:
            FileNotFoundError: If a bare name resolves to no file.
            ValueError: If a bare name resolves to multiple files.
            pydantic.ValidationError: If the YAML contents do not match
                the schema.
        """
        path = resolve_path(name_or_path, search_root=search_root)
        data = load_yaml(path)
        instance = cls.model_validate(data)
        object.__setattr__(instance, "_config_name", config_name or path.stem)
        return instance

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Validate a config from a nested dict.

        Args:
            data: Nested mapping matching the schema.

        Returns:
            A validated instance of ``cls``.
        """
        return cls.model_validate(data)

    @classmethod
    def from_flat_dict(cls, flat: dict[str, Any]) -> Self:
        """Validate a config from a flat dotted-key dict.

        Inverse of :meth:`to_flat_dict`. Useful for restoring a config from
        W&B / MLflow run params.

        Args:
            flat: Mapping where keys are dotted paths (e.g. ``"model.lr"``)
                and digit segments are interpreted as list indices.

        Returns:
            A validated instance of ``cls``.
        """
        return cls.model_validate(unflatten(flat))

    def update_from_args(
        self,
        args: Namespace | dict[str, Any],
        *,
        strict: bool = True,
        skip_none: bool = True,
    ) -> None:
        """Apply overrides from a pre-parsed argparse Namespace or plain dict.

        This is the **argparse-flavored** path: define your flags with
        :mod:`argparse`, parse them yourself, then hand the Namespace to
        onefig. Use this when you want full control over flag definitions,
        help text, and CLI ergonomics.

        For a quicker path that skips argparse entirely, see
        :meth:`update_from_cli`. Both routes share the same override engine
        (leaf-key resolution, dotted paths, ambiguity detection, Pydantic
        re-validation), so behavior is identical past the parsing step.

        Args:
            args: Either an :class:`argparse.Namespace` (whose ``__dict__``
                is read) or a plain mapping of override keys to values. Keys
                may be leaf names (e.g. ``"lr"``) or full dotted paths
                (e.g. ``"model.lr"``).
            strict: If ``True`` (default), unknown keys raise. If ``False``,
                unknown keys are silently ignored.
            skip_none: If ``True`` (default), entries whose value is ``None``
                are not applied — useful when argparse defaults are ``None``
                so unspecified flags don't clobber YAML values.

        Raises:
            ValueError: If a leaf key is ambiguous, or (when ``strict``) if
                a key is unknown.
            pydantic.ValidationError: If a value fails type validation at
                the destination field.
        """
        if isinstance(args, Namespace):
            raw = vars(args)
        else:
            raw = dict(args)
        if skip_none:
            raw = {k: v for k, v in raw.items() if v is not None}
        apply_overrides(self, raw, strict=strict)

    def update_from_cli(
        self,
        args: list[str] | None = None,
        *,
        strict: bool = True,
        exit_on_show: bool = True,
        exit_on_help: bool = True,
    ) -> None:
        """Apply ``key=value`` overrides parsed directly from CLI tokens.

        This is the **argparse-free** path: tokens are read straight from
        :data:`sys.argv` (or the list you pass) and parsed in place. Use this
        when you don't want to maintain an argparse parser alongside your
        config.

        For the argparse-flavored alternative, see :meth:`update_from_args`.
        Both routes share the same override engine (leaf-key resolution,
        dotted paths, ambiguity detection, Pydantic re-validation), so
        behavior is identical past the parsing step.

        Each token is of the form ``key=value`` where the key may be a leaf
        name or a full dotted path. Values are best-effort JSON-coerced
        (numbers, bools, ``null``, lists, dicts) and fall back to strings.

        Special tokens (consumed before override parsing):
          * ``--help`` / ``-h`` — print a schema-aware listing of every
            overridable field (type, default, current value, docstring) and
            exit. Short-circuits override application.
          * ``--show`` — apply overrides, then print the resolved config and
            exit.

        Args:
            args: List of CLI tokens. Defaults to ``sys.argv[1:]``.
            strict: If ``True`` (default), unknown keys raise.
            exit_on_show: If ``True`` (default), ``--show`` triggers
                :func:`sys.exit` after printing. Set to ``False`` to keep
                the call returning normally (useful in tests).
            exit_on_help: If ``True`` (default), ``--help`` / ``-h``
                triggers :func:`sys.exit` after printing. Set to ``False``
                to keep the call returning normally (useful in tests).

        Raises:
            ValueError: For malformed tokens, ambiguous leaf keys, or (when
                ``strict``) unknown keys.
            pydantic.ValidationError: If a value fails type validation at
                the destination field.
            SystemExit: If ``--show`` or ``--help`` / ``-h`` was passed and
                the corresponding ``exit_on_*`` flag is ``True``.
        """
        if args is None:
            args = sys.argv[1:]
        tokens = list(args)

        if "--help" in tokens or "-h" in tokens:
            self.print_help()
            if exit_on_help:
                sys.exit(0)
            return

        show_requested = "--show" in tokens
        tokens = [t for t in tokens if t != "--show"]

        overrides = parse_overrides(tokens)
        apply_overrides(self, overrides, strict=strict)

        if show_requested:
            self.display()
            if exit_on_show:
                sys.exit(0)

    def format_help(self, title: str | None = None) -> str:
        """Render this config's schema as a help string.

        Lists every overridable field with its dotted path, type, default,
        current value, and docstring (if any). Useful for argparse-driven
        scripts that want to surface the same information as the
        ``--help`` flag handled by :meth:`update_from_cli`.

        Args:
            title: Header for the output. Defaults to
                :attr:`config_name` (if set) or the class name.

        Returns:
            A multi-line help string ready to print.
        """
        return format_help(self, title=title or self._config_name)

    def print_help(self, title: str | None = None) -> None:
        """Print :meth:`format_help` to stdout."""
        print(self.format_help(title=title))

    def to_dict(self) -> dict[str, Any]:
        """Dump this config to a nested ``dict``.

        Returns:
            A nested mapping equivalent to :meth:`pydantic.BaseModel.model_dump`.
        """
        return self.model_dump()

    def to_flat_dict(self) -> dict[str, Any]:
        """Dump this config to a flat dotted-key ``dict``.

        Useful for hyperparameter logging into W&B / MLflow / TensorBoard.

        Returns:
            Mapping of ``"a.b.c"`` style keys to scalar values. Lists are
            indexed (``"tags.0"``, ``"layers.0.size"``).
        """
        return flatten(self.model_dump())

    def save_yaml(self, path: str | Path) -> None:
        """Serialize this config to YAML.

        Useful for snapshotting the resolved+overridden config alongside an
        experiment artifact. The resulting file round-trips back through
        :meth:`load`.

        Args:
            path: Filesystem path to write to.
        """
        cfg = OmegaConf.create(self.model_dump())
        OmegaConf.save(cfg, str(path))

    def freeze(self) -> Self:
        """Recursively freeze this config and all nested ConfigModels.

        After freezing, any attempt to mutate fields (or rename via
        :attr:`config_name`) raises :class:`FrozenConfigError`.

        Returns:
            ``self`` so the call can be chained.
        """
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, ConfigModel):
                value.freeze()
        object.__setattr__(self, "_frozen", True)
        return self

    @property
    def is_frozen(self) -> bool:
        """Whether this config is currently frozen.

        Returns:
            ``True`` if :meth:`freeze` has been called on this instance.
        """
        return self._frozen

    def display(self, name: str | None = None) -> None:
        """Print this config as an ASCII tree to stdout.

        Args:
            name: Title for the root of the tree. Defaults to
                :attr:`config_name` if set, else ``"Config"``.
        """
        title = name or self._config_name or "Config"
        print(format_tree(self.model_dump(), name=title))

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise FrozenConfigError(
                f"Cannot set {name!r}: config is frozen. "
                "Create a new instance via .from_dict() or .model_copy()."
            )

        cls = type(self)
        if (
            name in cls.model_fields
            or name in cls.__private_attributes__
            or name.startswith("__")
        ):
            super().__setattr__(name, value)
            return

        paths = _resolve_keys(self).get(name)
        if paths is None:
            super().__setattr__(name, value)
            return
        if len(paths) > 1:
            raise AttributeError(
                f"Ambiguous attribute {name!r}: matches {', '.join(paths)}. "
                "Use the full dotted path."
            )
        _set_dotted(self, paths[0], value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            # Defer to Pydantic for private-attr / dunder lookup.
            return super().__getattr__(name)  # type: ignore[misc]
        paths = _resolve_keys(self).get(name)
        if paths is None:
            raise AttributeError(name)
        if len(paths) > 1:
            raise AttributeError(
                f"Ambiguous attribute {name!r}: matches {', '.join(paths)}. "
                "Use the full dotted path."
            )
        obj: Any = self
        for part in paths[0].split("."):
            obj = getattr(obj, part)
        return obj
