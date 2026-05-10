from __future__ import annotations

import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_commit_hash() -> str | None:
    """Return the current ``git`` HEAD hash, or ``None`` if unavailable.

    Best-effort: any failure (no git binary, not a repo, timeout, etc.)
    returns ``None`` rather than raising. The result is cached for the
    process lifetime since the running code's commit does not change.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
