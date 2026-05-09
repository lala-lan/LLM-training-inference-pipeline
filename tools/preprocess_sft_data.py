# coding=utf-8
"""Optional HF tokenizer preprocessing (writes a datasets.Dataset to disk).

Legacy Megatron mmap binaries are not required for the Windows HF pipeline in ``training/``.
This script supports `step1_supervised_finetuning/data_process_scripts/data_process.sh`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def _fmt_messages_row(tok, prompt_type: str, row: dict) -> str:
    if "messages" in row and row["messages"]:
        return tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    inst = row.get("instruction") or row.get("instr") or ""
    inp = row.get("input") or ""
    out = row.get("output") or row.get("response") or ""
    if prompt_type == "alpaca":
        if inp:
            return (
                "Below is an instruction that describes a task. Write a response.\n\n"
                f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            )
        return (
            "Below is an instruction that describes a task. Write a response.\n\n"
            f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        )
    user = inst if not inp else f"{inst}\n{inp}"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return f"User:\n{user}\n\nAssistant:\n{out}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--tokenizer-name-or-path", type=str, required=True)
    ap.add_argument("--output-prefix", type=str, required=True)
    ap.add_argument("--task", type=str, default="sft")
    ap.add_argument("--tokenizer-type", type=str, default="PretrainedFromHF")
    ap.add_argument("--dataset-impl", type=str, default="mmap")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--question_max_length", type=int, default=10000)
    ap.add_argument("--question_truncation_side", type=str, default="right")
    ap.add_argument("--model_type", type=str, default=None)
    ap.add_argument("--prompt_type", type=str, default="chatml")
    args = ap.parse_args()

    files = [p.strip() for p in args.input.replace(";", ",").split(",") if p.strip()]
    ds = load_dataset("json", data_files=files, split="train")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = min(int(args.question_max_length), getattr(tokenizer, "model_max_length", 8192) or 8192)

    def _tok(batch):
        texts = []
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        cols = {k: batch[k] for k in keys}
        for i in range(n):
            row = {k: cols[k][i] for k in keys}
            texts.append(_fmt_messages_row(tokenizer, args.prompt_type, row))
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = [list(x) for x in enc["input_ids"]]
        return enc

    tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names, num_proc=max(1, args.workers))

    out = Path(args.output_prefix)
    out.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(out))
    meta = {
        "task": args.task,
        "tokenizer": args.tokenizer_name_or_path,
        "model_type": args.model_type,
        "prompt_type": args.prompt_type,
        "truncation_side": args.question_truncation_side,
    }
    (out / "preprocess_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
