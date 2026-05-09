import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def _read_config(model_dir: str) -> Optional[dict]:
    p = Path(model_dir) / "config.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_model_type(model_dir: str) -> Optional[str]:
    """Map HF config.json to pipeline family names used for prompts / inference."""
    cfg = _read_config(model_dir)
    if not cfg:
        return None
    arch = (cfg.get("architectures") or [None])[0]
    model_type = (cfg.get("model_type") or "").lower()
    name = str(arch or "") + str(cfg)

    if "Qwen3" in str(arch):
        return "qwen2.5"
    if "Qwen2_5" in str(arch) or "qwen2.5" in model_type:
        if "r1" in model_type or "deepseek" in name.lower():
            return "qwen2.5-R1"
        return "qwen2.5"
    if "Qwen2" in str(arch) or model_type == "qwen2":
        return "qwen2"
    if "Qwen1" in str(arch) or "qwen1" in model_type:
        return "qwen1.5"
    if "QWen" in str(arch) or "qwen" == model_type:
        return "qwen"
    if "Llama" in str(arch) or "llama" in model_type:
        return "llama2"
    if "ChatGLM3" in str(arch) or "chatglm3" in model_type:
        return "chatglm3"
    if "ChatGLM" in str(arch) or "chatglm2" in model_type:
        return "chatglm2"
    if "Baichuan" in str(arch) or "baichuan" in model_type:
        return "balchuan2"

    if arch:
        return "llama2"
    return None


def get_model_level(model_dir: str) -> str:
    cfg = _read_config(model_dir) or {}
    lvl = cfg.get("model_level") or cfg.get("pipeline_model_level")
    if lvl in ("lite", "std", "pro"):
        return lvl
    hidden = cfg.get("hidden_size")
    if isinstance(hidden, int) and hidden <= 2048:
        return "lite"
    if isinstance(hidden, int) and hidden >= 8192:
        return "pro"
    return "std"


class CommonTimer:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        if self.log_path.parent and not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        isfin: bool,
        phase: str,
        status: str,
        is_total_start: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        row = {
            "ts": time.time(),
            "phase": phase,
            "status": status,
            "isfin": isfin,
            "is_total_start": is_total_start,
        }
        if extra:
            row.update(extra)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)
