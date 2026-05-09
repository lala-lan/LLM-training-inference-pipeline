from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.utils import get_pgu_type  # noqa: E402

_ = get_pgu_type(str(REPO_ROOT))

try:
    from pipeline.dpo_running import dpo_run  # noqa: E402
    from pipeline.pretrain_running import pretrain_run  # noqa: E402
    from pipeline.sft_lora_running import sft_lora_run  # noqa: E402
    from pipeline.sft_running import sft_run  # noqa: E402
except ImportError:  # pragma: no cover - `python pipeline/train_master.py`
    from dpo_running import dpo_run  # type: ignore  # noqa: E402
    from pretrain_running import pretrain_run  # type: ignore  # noqa: E402
    from sft_lora_running import sft_lora_run  # type: ignore  # noqa: E402
    from sft_running import sft_run  # type: ignore  # noqa: E402


def load_params(arg: str) -> dict:
    p = Path(arg)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        decoded = base64.b64decode(arg).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return json.loads(arg)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.train_master <params.json | base64-json>")
        raise SystemExit(2)
    raw = sys.argv[1]
    params = load_params(raw)
    print(f"task={params.get('task')!r}")

    task = params.get("task")
    if task in ("train", "test"):
        sft_run(params)
    elif task in ("lora_train", "lora_test"):
        sft_lora_run(params)
    elif task in ("dpo_train", "dpo_test"):
        dpo_run(params)
    elif task in ("pretrain", "pretrain_test"):
        pretrain_run(params)
    else:
        print(f"## error: no task named {task!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
