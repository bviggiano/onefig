from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

YAML_EXTS = (".yaml", ".yml")


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

    Args:
        path: Path to a YAML file.

    Returns:
        A plain nested ``dict`` with all ``${...}`` interpolations resolved.
    """
    path = Path(path)
    logger.debug("Loading config from %s", path)
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


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
