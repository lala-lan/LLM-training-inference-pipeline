from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.utils_entropy import run_python_module
from step1_supervised_finetuning.modify import CommonTimer, get_model_type

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_export(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def dpo_run(params: dict) -> None:
    task = params.get("task", "")
    if "dpo_train" in task:
        run_steps = [1, 2, 3]
    elif "dpo_test" in task:
        run_steps = [4]
    else:
        print(f"task:{task} should be 'dpo_train'/'dpo_test'")
        raise SystemExit(1)

    load_model_path = params["base_model_path"]
    work_path = Path(params["work_path"])
    work_path.mkdir(parents=True, exist_ok=True)

    output_model_dir = work_path / "output_origin_model"
    trans_output = work_path / "trans_hf"
    mmap_dir = work_path / "process_data"
    inference_out = work_path / "inference"

    if "dpo_train" in task:
        model_type = get_model_type(load_model_path)
    else:
        model_type = get_model_type(str(trans_output))
    if model_type is None:
        print(f"Unsupported model path: {load_model_path}")
        raise SystemExit(1)

    progress_log = params["progress_log_file_path"]
    timer = CommonTimer(progress_log)
    timer.log(isfin=False, phase="start", status="start", is_total_start=True)

    if 1 in run_steps:
        timer.log(isfin=False, phase="data_process", status="running")
        mmap_dir.mkdir(parents=True, exist_ok=True)
        train_data = params["train_data_path"]
        if isinstance(train_data, list):
            train_data = ";".join(train_data)
        _write_json(
            mmap_dir / "dpo_dataset_manifest.json",
            {"train_data_path": train_data, "backend": "hf_jsonl_dpo"},
        )
        timer.log(isfin=True, phase="data_process", status="running")

    train_payload = {
        "base_model_path": load_model_path,
        "reference_model_path": params.get("reference_model_path", load_model_path),
        "output_model_dir": str(output_model_dir),
        "train_data_path": params["train_data_path"],
        "dev_data_path": params.get("dev_data_path"),
        "dev_split_rate": float(params.get("dev_split_rate", 0.0)),
        "batch_size": int(params.get("batch_size", 1)),
        "epochs": float(params.get("epochs", 1)),
        "learning_rate": float(params.get("learning_rate", 1e-6)),
        "max_seq_len": int(params.get("max_seq_len", 2048)),
        "max_prompt_length": int(params.get("max_prompt_length", 512)),
        "max_target_length": int(params.get("max_target_length", 1536)),
        "lr_scheduler_type": params.get("lr_scheduler_type", "cosine"),
        "num_warmup_steps": int(params.get("num_warmup_steps", 0)),
        "eval_steps": int(params.get("eval_steps", 200)),
        "save_steps": int(params.get("save_steps", 500)),
        "logging_steps": int(params.get("logging_steps", 10)),
        "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 1)),
        "gradient_checkpointing": bool(params.get("gradient_checkpointing", True)),
        "save_total_limit": int(params.get("save_total_limit", 3)),
        "dpo_beta": float(params.get("dpo_beta", 0.1)),
        "dpo_loss_type": params.get("dpo_loss_type", "sigmoid"),
    }

    if 2 in run_steps:
        timer.log(isfin=False, phase="train", status="running")
        cfg_path = work_path / "_pipeline_dpo_params.json"
        _write_json(cfg_path, train_payload)
        code = run_python_module(str(REPO_ROOT), "training.run_dpo", ["--params", str(cfg_path)])
        if code != 0:
            timer.log(isfin=True, phase="train", status="fail")
            raise SystemExit(code)
        timer.log(isfin=True, phase="train", status="running")

    merged = output_model_dir / "hf_merged"
    if 3 in run_steps:
        timer.log(isfin=False, phase="convert_model", status="running")
        if not merged.is_dir():
            print("Missing merged HF checkpoint:", merged)
            timer.log(isfin=True, phase="convert_model", status="fail")
            raise SystemExit(1)
        _sync_export(merged, trans_output)
        timer.log(isfin=True, phase="convert_model", status="running")

    if 4 in run_steps:
        timer.log(isfin=False, phase="inference", status="running")
        inference_out.mkdir(parents=True, exist_ok=True)
        infer_payload = {
            "data_file": params["test_file"],
            "model_name_or_path": str(trans_output),
            "tokenize_name_or_path": str(trans_output),
            "output": str(inference_out),
            "output_file": params["test_output_file"],
            "prompt_type": params.get("prompt_type", "chatml"),
            "generate_configs": params.get("generate_configs", {}) or {},
            "model_type": model_type,
        }
        ip = work_path / "_pipeline_infer_params.json"
        _write_json(ip, infer_payload)
        code = run_python_module(str(REPO_ROOT), "training.run_inference", ["--params", str(ip)])
        if code != 0:
            timer.log(isfin=True, phase="inference", status="fail")
            raise SystemExit(code)
        timer.log(isfin=True, phase="inference", status="running")

    timer.log(isfin=True, phase="end", status="success")
