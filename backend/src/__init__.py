"""Backend package root.

Auto-loads the project-root `.env` file the first time anything inside `src`
is imported, so scripts and notebooks don't need to source it manually.
Walks upward from this file until it finds a `.env` — that way both
`cd backend && python scripts/foo.py` and `python backend/scripts/foo.py`
behave identically.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv not installed yet — silent no-op
    load_dotenv = None  # type: ignore[assignment]


def _autoload_env() -> None:
    if load_dotenv is None:
        return
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


_autoload_env()
