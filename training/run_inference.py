from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _prompt_from_row(tok, prompt_type: str, row: dict) -> str:
    if "messages" in row and row["messages"]:
        return tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    inst = row.get("instruction") or row.get("instr") or ""
    inp = row.get("input") or ""
    if prompt_type == "alpaca":
        if inp:
            return (
                "Below is an instruction that describes a task. Write a response.\n\n"
                f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. Write a response.\n\n"
            f"### Instruction:\n{inst}\n\n### Response:\n"
        )
    user = inst if not inp else f"{inst}\n{inp}"
    messages = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User:\n{user}\n\nAssistant:\n"


def run_from_params(p: dict) -> None:
    model_path = p["model_name_or_path"]
    tok_path = p.get("tokenize_name_or_path") or model_path
    data_file = Path(p["data_file"])
    out_dir = Path(p["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(p["output_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    gen_cfg = p.get("generate_configs") or {}
    if isinstance(gen_cfg, str):
        gen_cfg = json.loads(gen_cfg)

    prompt_type = p.get("prompt_type", "chatml")
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))

    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True, trust_remote_code=True)
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
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows_out = []
    with data_file.open(encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in tqdm(lines, desc="inference"):
        row = json.loads(line)
        prompt = _prompt_from_row(tokenizer, prompt_type, row)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        row_out = dict(row)
        row_out["prediction"] = gen.strip()
        rows_out.append(row_out)

    with out_file.open("w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default=None)
    args = ap.parse_args()
    if not args.params:
        raise SystemExit("--params JSON path is required for python -m training.run_inference")
    p = json.loads(Path(args.params).read_text(encoding="utf-8"))
    run_from_params(p)


if __name__ == "__main__":
    main()
