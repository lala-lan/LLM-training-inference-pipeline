from __future__ import annotations

import os
import subprocess
from typing import Optional


def get_pgu_type(root_path: Optional[str] = None) -> str:
    """Return coarse GPU label for optional branching (A800 vs OTHER)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return "OTHER"
        name = r.stdout.strip().splitlines()[0]
        if "A800" in name or "A100" in name:
            return "A800"
        return "OTHER"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "OTHER"
