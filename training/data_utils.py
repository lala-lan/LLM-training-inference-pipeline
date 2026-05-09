from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def _expand_paths(spec: str) -> List[str]:
    parts = [p.strip() for p in spec.replace(",", ";").split(";") if p.strip()]
    out: List[str] = []
    for p in parts:
        path = Path(p)
        if path.is_file():
            out.append(str(path.resolve()))
        elif path.is_dir():
            for ext in ("*.jsonl", "*.json"):
                out.extend(str(x) for x in sorted(path.glob(ext)))
    return out


def iter_jsonl(paths: List[str]) -> Iterator[dict]:
    for fp in paths:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def load_sft_dataset(train_spec: str, eval_spec: str | None, eval_split: float):
    files = _expand_paths(train_spec)
    if not files:
        raise FileNotFoundError(f"No training files for spec: {train_spec}")
    ds = load_dataset("json", data_files=files, split="train")

    if eval_spec:
        ef = _expand_paths(eval_spec)
        if ef:
            eval_ds = load_dataset("json", data_files=ef, split="train")
            return DatasetDict({"train": ds, "eval": eval_ds})

    if eval_split and 0 < eval_split < 1:
        split = ds.train_test_split(test_size=eval_split)
        return DatasetDict({"train": split["train"], "eval": split["test"]})
    return DatasetDict({"train": ds})


def load_dpo_dataset(train_spec: str, eval_spec: str | None, eval_split: float):
    files = _expand_paths(train_spec)
    if not files:
        raise FileNotFoundError(f"No DPO training files for spec: {train_spec}")
    ds = load_dataset("json", data_files=files, split="train")

    if eval_spec:
        ef = _expand_paths(eval_spec)
        if ef:
            eval_ds = load_dataset("json", data_files=ef, split="train")
            return DatasetDict({"train": ds, "eval": eval_ds})

    if eval_split and 0 < eval_split < 1:
        split = ds.train_test_split(test_size=eval_split)
        return DatasetDict({"train": split["train"], "eval": split["test"]})
    return DatasetDict({"train": ds})


def load_pretrain_dataset(train_spec: str, eval_spec: str | None, eval_split: float):
    files = _expand_paths(train_spec)
    if not files:
        raise FileNotFoundError(f"No pretrain files for spec: {train_spec}")

    json_files = [f for f in files if f.lower().endswith((".json", ".jsonl"))]
    txt_files = [f for f in files if f.lower().endswith(".txt")]
    chunks = []
    if json_files:
        chunks.append(load_dataset("json", data_files=json_files, split="train"))
    for fp in txt_files:
        text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        chunks.append(Dataset.from_list([{"text": text}]))

    if not chunks:
        raise ValueError("Pretrain expects .json/.jsonl (with a `text` field) or .txt files.")
    ds = concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]
    if "text" not in ds.column_names:
        raise ValueError("Pretrain JSON must include a `text` column.")

    if eval_spec:
        ef = _expand_paths(eval_spec)
        if ef:
            ej = [f for f in ef if f.lower().endswith((".json", ".jsonl"))]
            et = [f for f in ef if f.lower().endswith(".txt")]
            ev_chunks = []
            if ej:
                ev_chunks.append(load_dataset("json", data_files=ej, split="train"))
            for fp in et:
                t = Path(fp).read_text(encoding="utf-8", errors="ignore")
                ev_chunks.append(Dataset.from_list([{"text": t}]))
            eval_ds = concatenate_datasets(ev_chunks) if len(ev_chunks) > 1 else ev_chunks[0]
            return DatasetDict({"train": ds, "eval": eval_ds})

    if eval_split and 0 < eval_split < 1:
        split = ds.train_test_split(test_size=eval_split)
        return DatasetDict({"train": split["train"], "eval": split["test"]})
    return DatasetDict({"train": ds})
