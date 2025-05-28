#!/usr/bin/env python3
"""
Merge affiliations from PDF papers into an existing JSONL list and flag the
records that mention a preferred company (OpenAI, Google, …).

Optimized to:
- Only extract the first N characters (default 500–1000) from the first page.
- Avoid reading and tokenizing full pages.
"""

import os
import json
import argparse
import re
import pdfplumber
import warnings
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings(
    "ignore",
    message=r"CropBox missing from /Page.*",
    module="pdfplumber",
)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge affiliations and select preferred-company papers",
    )
    p.add_argument("papers_dir",     help="Directory containing PDFs (e.g. XXX_pdf)")
    p.add_argument("existing_jsonl", help="Path to existing JSONL file")
    return p.parse_args()

def load_model():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        temperature=0.6,
        max_new_tokens=2048,
        return_full_text=False,
    )

def extract_text(path: str, max_chars: int = 1000) -> str:
    """
    Extract up to `max_chars` characters from the first page of a PDF.
    """
    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            return ""
        first_page = pdf.pages[0]
        text = first_page.extract_text() or ""
        return text[:max_chars]

def extract_affiliations(gen, text: str) -> list[str]:
    """
    Call the LLM twice (reason, then extract) and return a deduplicated list
    of institution names.
    """
    think_prompt = (
        "You are an assistant that extracts author institutions from scientific papers.\n"
        "First, think step by step about which institutions appear in the snippet below, "
        "and then at the end clearly list your conclusions.\n\n"
        f"Snippet:\n{text[:500]}\n\n"
        "Your reasoning and conclusion:\n"
    )
    reasoning = gen(think_prompt)[0]["generated_text"]

    extract_prompt = (
        "Now produce the final answer.\n\n"
        "***Output requirements***\n"
        "• Format: a *valid* JSON array of strings, e.g.\n"
        '  ["MIT","Google DeepMind"]\n'
        "• Each element must be a plain string.\n"
        "• Return an empty array (`[]`) if none are found.\n"
        "• Do **not** wrap the JSON in markdown or add any other text.\n\n"
        f"{reasoning}\n\n"
        "JSON array:"
    )
    output = gen(extract_prompt)[0]["generated_text"]

    try:
        start, end = output.index("["), output.rindex("]") + 1
        raw = json.loads(output[start:end])
    except Exception:
        raw = [ln.strip("-• ").strip() for ln in output.splitlines() if ln.strip()]

    flat: list[str] = []
    def _flatten(item):
        if isinstance(item, str):
            flat.append(item.strip())
        elif isinstance(item, dict):
            flat.extend(str(v).strip() for v in item.values())
        elif isinstance(item, (list, tuple, set)):
            for sub in item:
                _flatten(sub)
        elif item is not None:
            flat.append(str(item).strip())
    _flatten(raw)

    seen, dedup = set(), []
    for inst in flat:
        key = inst.lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(inst)
    return dedup

_companies = [
    r"open[\s\-]*ai",
    r"meta",
    r"google",
    r"deep[\s\-]*mind",
    r"anthropic",
    r"microsoft",
    r"deep[\s\-]*seek",
]
_pattern = re.compile(r"(?i)\b(?:" + "|".join(_companies) + r")\b")
def matches_company(institutions: list[str]) -> bool:
    return any(_pattern.search(inst) for inst in institutions)

def main():
    args = parse_args()

    folder = os.path.basename(os.path.normpath(args.papers_dir))
    prefix = folder[:-4] if folder.endswith("_pdf") else folder
    merged_path = f"{prefix}_affiliation.jsonl"
    chosen_path = f"{prefix}_chosen_affiliation.jsonl"

    existing: dict[str, dict] = {}
    with open(args.existing_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            rid = rec.get("id")
            if rid and rid not in existing:
                existing[rid] = rec

    print("Loading model…")
    gen = load_model()

    for fname in sorted(os.listdir(args.papers_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        pid  = os.path.splitext(fname)[0]
        path = os.path.join(args.papers_dir, fname)
        print(f"Processing {fname}…")

        full_txt = extract_text(path, max_chars=1000)

        try:
            insts = extract_affiliations(gen, full_txt)
        except Exception as e:
            logging.exception(f"Affiliation extraction failed for {fname}: {e}")
            insts = []

        if pid in existing:
            rec    = existing[pid]
            merged = list(dict.fromkeys(rec.get("institutions", []) + insts))
            rec["institutions"] = merged
            rec["preference"]   = 1 if matches_company(merged) else 0
        else:
            print(f"  Warning: {pid} not found in existing JSONL")

    # Write merged
    with open(merged_path, "w", encoding="utf-8") as fout:
        for rec in existing.values():
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote merged to {merged_path}")

    # Write chosen subset
    with open(chosen_path, "w", encoding="utf-8") as fout:
        for rec in existing.values():
            if rec.get("preference") == 1:
                out = {k: rec.get(k) for k in ("id", "pdf", "title", "authors", "institutions")}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote chosen to {chosen_path}")


if __name__ == "__main__":
    main()
