from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from training.data_utils import load_sft_dataset


def _fmt_row(tok, prompt_type: str, row: dict) -> str:
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
    ap.add_argument("--params", type=str, required=True)
    args = ap.parse_args()
    p = json.loads(Path(args.params).read_text(encoding="utf-8"))

    base = p["base_model_path"]
    out_dir = Path(p["output_model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_spec = p["train_data_path"]
    if isinstance(train_spec, list):
        train_spec = ";".join(train_spec)
    dev_spec = p.get("dev_data_path")
    if isinstance(dev_spec, list):
        dev_spec = ";".join(dev_spec)
    eval_split = float(p.get("dev_split_rate", 0.0))

    raw = load_sft_dataset(train_spec, dev_spec or None, eval_split)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_type = p.get("prompt_type", "chatml")

    def _map(batch):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        cols = {k: batch[k] for k in keys}
        texts = []
        for i in range(n):
            row = {k: cols[k][i] for k in keys}
            texts.append(_fmt_row(tokenizer, prompt_type, row))
        return {"text": texts}

    tr_cols = list(raw["train"].column_names)
    train_ds = raw["train"].map(_map, batched=True, remove_columns=tr_cols)
    if "eval" in raw:
        ev_cols = list(raw["eval"].column_names)
        eval_ds = raw["eval"].map(_map, batched=True, remove_columns=ev_cols)
    else:
        eval_ds = None

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    load_kw = dict(trust_remote_code=True)
    if torch.cuda.is_available():
        load_kw["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    if bool(p.get("gradient_checkpointing", True)):
        model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(p.get("batch_size", 1)),
        per_device_eval_batch_size=int(p.get("batch_size", 1)),
        gradient_accumulation_steps=int(p.get("gradient_accumulation_steps", 1)),
        learning_rate=float(p.get("learning_rate", 2e-5)),
        num_train_epochs=float(p.get("epochs", 1)),
        lr_scheduler_type=p.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(p.get("num_warmup_steps", 0)),
        logging_steps=int(p.get("logging_steps", 10)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(p.get("eval_steps", 200)) if eval_ds is not None else None,
        save_steps=int(p.get("save_steps", 500)),
        save_total_limit=int(p.get("save_total_limit", 3)),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(p.get("gradient_checkpointing", True)),
        report_to=[],
    )

    max_len = int(p.get("max_seq_len", 2048))
    sig = inspect.signature(SFTTrainer.__init__)
    kw = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
    )
    if "max_seq_length" in sig.parameters:
        kw["max_seq_length"] = max_len
    elif "max_seq_len" in sig.parameters:
        kw["max_seq_len"] = max_len
    if "processing_class" in sig.parameters:
        kw["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kw["tokenizer"] = tokenizer
    trainer = SFTTrainer(**kw)

    trainer.train()
    final_dir = out_dir / "hf_merged"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    (out_dir / "latest").write_text("hf_merged\n", encoding="utf-8")


if __name__ == "__main__":
    main()
