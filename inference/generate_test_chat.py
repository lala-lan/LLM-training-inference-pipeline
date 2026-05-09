from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--excute_path", type=str, default=".")
    ap.add_argument("--data_file", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="qwen2.5")
    ap.add_argument("--tokenize_name_or_path", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--output_file", type=str, required=True)
    ap.add_argument("--prompt_type", type=str, default="chatml")
    ap.add_argument("--generate_configs", type=str, default="{}")
    ap.add_argument("--process_log", type=str, default=None)
    args = ap.parse_args()

    try:
        gen_cfg = json.loads(args.generate_configs)
    except json.JSONDecodeError:
        gen_cfg = {}

    from training.run_inference import run_from_params

    run_from_params(
        {
            "data_file": args.data_file,
            "model_name_or_path": args.model_name_or_path,
            "tokenize_name_or_path": args.tokenize_name_or_path or args.model_name_or_path,
            "output": args.output,
            "output_file": args.output_file,
            "prompt_type": args.prompt_type,
            "generate_configs": gen_cfg,
            "model_type": args.model_type,
            "gpus": args.gpus,
            "process_log": args.process_log,
        }
    )


if __name__ == "__main__":
    main()
