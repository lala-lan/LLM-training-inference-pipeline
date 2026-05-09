from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

from training.data_utils import load_dpo_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, required=True)
    args = ap.parse_args()
    p = json.loads(Path(args.params).read_text(encoding="utf-8"))

    base = p["base_model_path"]
    ref = p.get("reference_model_path") or base
    out_dir = Path(p["output_model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_spec = p["train_data_path"]
    if isinstance(train_spec, list):
        train_spec = ";".join(train_spec)
    dev_spec = p.get("dev_data_path")
    if isinstance(dev_spec, list):
        dev_spec = ";".join(dev_spec)
    eval_split = float(p.get("dev_split_rate", 0.0))

    dsets = load_dpo_dataset(train_spec, dev_spec or None, eval_split)
    train_ds = dsets["train"]
    eval_ds = dsets["eval"] if "eval" in dsets else None

    for col in ("prompt", "chosen", "rejected"):
        if col not in train_ds.column_names:
            raise ValueError(f"DPO dataset must contain column '{col}'")

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    load_kw = dict(trust_remote_code=True)
    if torch.cuda.is_available():
        load_kw["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    ref_model = AutoModelForCausalLM.from_pretrained(ref, **load_kw)
    if bool(p.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        ref_model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(p.get("batch_size", 1)),
        per_device_eval_batch_size=int(p.get("batch_size", 1)),
        gradient_accumulation_steps=int(p.get("gradient_accumulation_steps", 1)),
        learning_rate=float(p.get("learning_rate", 1e-6)),
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

    beta = float(p.get("dpo_beta", 0.1))
    loss_type = p.get("dpo_loss_type", "sigmoid")
    max_len = int(p.get("max_seq_len", 2048))
    max_prompt = int(p.get("max_prompt_length", max_len // 2))
    max_tgt = int(p.get("max_target_length", max_len - max_prompt))

    sig = inspect.signature(DPOTrainer.__init__)
    kw = dict(
        model=model,
        ref_model=ref_model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        beta=beta,
        loss_type=loss_type,
        max_prompt_length=max_prompt,
        max_length=max_len,
        max_target_length=max_tgt,
    )
    if "processing_class" in sig.parameters:
        kw["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kw["tokenizer"] = tokenizer
    kw = {k: v for k, v in kw.items() if k in sig.parameters}

    trainer = DPOTrainer(**kw)

    trainer.train()
    final_dir = out_dir / "hf_merged"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (out_dir / "latest").write_text("hf_merged\n", encoding="utf-8")


if __name__ == "__main__":
    main()
