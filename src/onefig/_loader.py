from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)

YAML_EXTS = (".yaml", ".yml")
EXTENDS_KEY = "extends"


def resolve_path(
    name_or_path: str | Path,
    search_root: str | Path | None = None,
) -> Path:
    """Resolve a YAML config name or path to a concrete :class:`~pathlib.Path`.

    Args:
        name_or_path: Either an absolute or existing relative path to a YAML
            file, or a bare config name to search for under ``search_root``.
        search_root: Directory under which to recursively search when
            ``name_or_path`` is a bare name. Defaults to the current working
            directory.

    Returns:
        The resolved path to a YAML file on disk.

    Raises:
        FileNotFoundError: If a bare name resolves to no file under the
            search root.
        ValueError: If a bare name resolves to multiple files (ambiguous).
    """
    p = Path(name_or_path)
    if p.is_file():
        return p
    root = Path(search_root) if search_root is not None else Path.cwd()
    return _find_by_name(str(name_or_path), root)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and resolve OmegaConf interpolations into a plain dict.

    Honors a top-level ``extends:`` key: if present, the referenced file(s)
    are loaded first and the current file's content is deep-merged on top.
    ``extends:`` accepts a single path or a list of paths; later entries
    in a list override earlier ones, and the current file always wins over
    its ancestors. Paths are resolved relative to the file containing the
    ``extends:`` key.

    Interpolations (``${other.key}``, ``${oc.env:VAR}``, ...) are resolved
    after the full chain has been merged, so cross-file references work
    naturally.

    Args:
        path: Path to a YAML file.

    Returns:
        A plain nested ``dict`` with the ``extends`` chain merged and all
        ``${...}`` interpolations resolved.

    Raises:
        FileNotFoundError: If an ``extends:`` target doesn't exist on disk.
        ValueError: If an ``extends:`` chain contains a cycle, or if the
            value of ``extends:`` isn't a string or list of strings.
    """
    path = Path(path)
    logger.debug("Loading config from %s", path)
    merged = _compose(path, _seen=set())
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def _compose(path: Path, *, _seen: set[Path]) -> DictConfig:
    """Recursively merge an ``extends:`` chain into a single DictConfig.

    Maintains a set of resolved absolute paths along the current chain to
    detect cycles. Interpolations are NOT resolved here — that happens
    once at the top of :func:`load_yaml`, after the entire chain has been
    merged, so cross-file references are sound.
    """
    resolved = path.resolve()
    if resolved in _seen:
        chain = " → ".join(str(p) for p in _seen) + f" → {resolved}"
        raise ValueError(f"Cycle in extends chain: {chain}")

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        # Top-level lists, scalars, etc. are not valid onefig configs; let
        # downstream validation produce the schema-aware error.
        return cfg  # type: ignore[return-value]

    parents_raw = cfg.pop(EXTENDS_KEY, None)
    if parents_raw is None:
        return cfg

    parent_paths = _resolve_extends(parents_raw, base_dir=path.parent, source=path)

    next_seen = _seen | {resolved}
    merged: Any = OmegaConf.create({})  # merge() widens to ListConfig | DictConfig
    for parent_path in parent_paths:
        parent = _compose(parent_path, _seen=next_seen)
        merged = OmegaConf.merge(merged, parent)
    return OmegaConf.merge(merged, cfg)  # type: ignore[return-value]


def _resolve_extends(value: Any, *, base_dir: Path, source: Path) -> list[Path]:
    """Turn the raw value of an ``extends:`` key into a list of resolved paths."""
    # `value` may be a ListConfig if it came from OmegaConf; collapse to a
    # plain Python list before type-checking.
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=False)

    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        items = list(value)
    else:
        raise ValueError(
            f"{source}: 'extends' must be a string or list of strings, "
            f"got {type(value).__name__}: {value!r}"
        )

    out: list[Path] = []
    for item in items:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{source}: 'extends' target not found: {item!r} "
                f"(resolved to {candidate})"
            )
        out.append(candidate)
    return out


def _find_by_name(name: str, root: Path) -> Path:
    targets = [name] if name.endswith(YAML_EXTS) else [name + ext for ext in YAML_EXTS]

    matches: list[Path] = []
    for target in targets:
        matches.extend(root.rglob(target))

    if not matches:
        raise FileNotFoundError(f"No YAML file matching {name!r} under {root}")
    if len(matches) > 1:
        joined = "\n  ".join(str(m) for m in matches)
        raise ValueError(f"Multiple YAML files matching {name!r}:\n  {joined}")
    return matches[0]
