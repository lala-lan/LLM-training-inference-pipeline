from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Union


Cmd = Union[str, List[str]]


def execute_cmd(cmd: Cmd, *, cwd: Optional[str] = None) -> int:
    """Run a shell command (str) or argument list. Windows-friendly."""
    kw = {}
    if cwd:
        kw["cwd"] = cwd
    if isinstance(cmd, list):
        return subprocess.run(cmd, **kw).returncode
    return subprocess.run(cmd, shell=True, **kw).returncode


def run_python_module(repo_root: str, module: str, module_args: List[str]) -> int:
    env = os.environ.copy()
    if os.name == "nt":
        # Chinese Windows defaults to GBK; TRL reads UTF-8 templates at import time.
        env["PYTHONUTF8"] = "1"
    sep = os.pathsep
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{sep}{prev}" if prev else repo_root
    return subprocess.run(
        [sys.executable, "-m", module, *module_args],
        cwd=repo_root,
        env=env,
    ).returncode
