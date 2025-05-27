#!/usr/bin/env python3
"""
Merge affiliations from PDF papers into an existing JSONL list and flag the
records that mention a preferred company (OpenAI, Google, …).

Key features
------------
✓ Robustly parses any mix of strings / dicts / nested lists from the LLM.
✓ Stricter output-format prompt so the model rarely misbehaves.
✓ Per-PDF try/except so one bad paper doesn’t abort the whole run.
✓ TOKENIZERS_PARALLELISM=false to silence fork warnings.
✓ Expanded company list with DeepSeek and more forgiving spacing/hyphen rules.
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

# 1) Silence CropBox warnings from pdfminer/pdfplumber
warnings.filterwarnings(
    "ignore",
    message=r"CropBox missing from /Page.*",
    module="pdfplumber",
)

# 2) Suppress lower-level pdfminer logs entirely
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# 3) Kill Hugging Face tokeniser fork spam
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge affiliations and select preferred-company papers",
    )
    p.add_argument("papers_dir",     help="Directory containing PDFs (e.g. XXX_pdf)")
    p.add_argument("existing_jsonl", help="Path to existing JSONL file")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# LLM loader
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(path: str, max_pages: int = 2, footer_margin: float = 50) -> str:
    """
    Extract text from the first `max_pages` pages and gather footer words
    from page 1 (for conference info, etc.).
    """
    pages_text, footer_text = [], []
    with pdfplumber.open(path) as pdf:
        # body
        for page in pdf.pages[:max_pages]:
            pages_text.append(page.extract_text() or "")
        # footer
        first = pdf.pages[0]
        h = first.height
        for w in first.extract_words():
            if float(w["bottom"]) >= h - footer_margin:
                footer_text.append(w["text"])
    combined = "\n".join(pages_text)
    if footer_text:
        combined += "\n" + " ".join(footer_text)
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Affiliation extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_affiliations(gen, text: str) -> list[str]:
    """
    Call the LLM twice (reason, then extract) and return a deduplicated list
    of institution names.
    """
    # 1) Chain-of-thought
    think_prompt = (
        "You are an assistant that extracts author institutions from scientific papers.\n"
        "First, think step by step about which institutions appear in the snippet below, "
        "and then at the end clearly list your conclusions.\n\n"
        f"Snippet:\n{text[:2000]}\n\n"
        "Your reasoning and conclusion:\n"
    )
    reasoning = gen(think_prompt)[0]["generated_text"]

    # 2) Strict extraction prompt
    extract_prompt = (
        "Now produce the final answer.\n\n"
        "***Output requirements***\n"
        "• Format: a *valid* JSON array of strings, e.g.\n"
        '  [\"MIT\",\"Google DeepMind\"]\n'
        "• Each element must be a plain string (no objects, no nested arrays).\n"
        "• Return an empty array (`[]`) if none are found.\n"
        "• Do **not** wrap the JSON in markdown or add any other text.\n\n"
        f"{reasoning}\n\n"
        "JSON array:"
    )
    output = gen(extract_prompt)[0]["generated_text"]

    # ── Robust post-processing ────────────────────────────────────────────
    try:
        start, end = output.index("["), output.rindex("]")
        raw = json.loads(output[start:end + 1])
    except Exception:
        # Model slipped up — treat each non-blank line as a candidate string
        raw = [ln.strip("-• ").strip() for ln in output.splitlines() if ln.strip()]

    flat: list[str] = []

    def _flatten(item):
        """Recursively flatten nested structures into plain strings."""
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

    # Deduplicate case-insensitively while preserving order
    seen, dedup = set(), []
    for inst in flat:
        key = inst.lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(inst)

    return dedup


# ──────────────────────────────────────────────────────────────────────────────
# Company filter
# ──────────────────────────────────────────────────────────────────────────────

_companies = [
    r"open[\s\-]*ai",     # OpenAI, Open AI, Open-AI
    r"meta",              # Meta
    r"google",            # Google
    r"deep[\s\-]*mind",   # DeepMind, Deep Mind, Deep-Mind
    r"anthropic",         # Anthropic
    r"microsoft",         # Microsoft
    # r"byte[\s\-]*dance",  # ByteDance, Byte Dance, Byte-Dance
    r"deep[\s\-]*seek",   # DeepSeek, Deep Seek, Deep-Seek   ← NEW
]

# match whole words, ignoring case, allowing spaces or hyphens inside
_pattern = re.compile(r"(?i)\b(?:" + "|".join(_companies) + r")\b")

def matches_company(institutions: list[str]) -> bool:
    return any(_pattern.search(inst) for inst in institutions)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # derive output filenames
    folder = os.path.basename(os.path.normpath(args.papers_dir))
    prefix = folder[:-4] if folder.endswith("_pdf") else folder
    merged_path = f"{prefix}_affiliation.jsonl"
    chosen_path = f"{prefix}_chosen_affiliation.jsonl"

    # ── Load existing JSONL ───────────────────────────────────────────────
    existing: dict[str, dict] = {}
    with open(args.existing_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line: {line[:30]}…")
                continue
            rid = rec.get("id")
            if rid and rid not in existing:
                existing[rid] = rec

    # ── Load LLM ──────────────────────────────────────────────────────────
    print("Loading model…")
    gen = load_model()

    # ── Process PDFs ──────────────────────────────────────────────────────
    for fname in sorted(os.listdir(args.papers_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        pid  = os.path.splitext(fname)[0]
        path = os.path.join(args.papers_dir, fname)
        print(f"Processing {fname}…")

        txt = extract_text(path)
        if not txt.strip():
            print(f"  Warning: no text extracted from {fname}")
            continue

        # Safe extraction
        try:
            insts = extract_affiliations(gen, txt)
        except Exception as e:
            logging.exception(f"Affiliation extraction failed for {fname}: {e}")
            insts = []

        # Fallback: layout-aware page 1 if nothing found
        if not insts:
            print("  No affiliations found — trying layout-aware fallback on page 1…")
            try:
                with pdfplumber.open(path) as pdf:
                    layout_txt = pdf.pages[0].extract_text(layout=True) or ""
                insts = extract_affiliations(gen, layout_txt)
            except Exception as e:
                logging.exception(f"Fallback extraction failed for {fname}: {e}")
                insts = []

        # Merge into existing record
        if pid in existing:
            rec = existing[pid]
            merged = list(dict.fromkeys(rec.get("institutions", []) + insts))
            rec["institutions"] = merged
            rec["preference"]   = 1 if matches_company(merged) else 0
        else:
            print(f"  Warning: {pid} not found in existing JSONL")

    # ── Write outputs ─────────────────────────────────────────────────────
    # 1) Merged file
    with open(merged_path, "w", encoding="utf-8") as fout:
        for rec in existing.values():
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote merged to {merged_path}")

    # 2) Chosen (company-matching) subset
    with open(chosen_path, "w", encoding="utf-8") as fout:
        for rec in existing.values():
            if rec.get("preference") == 1:
                out = {k: rec.get(k) for k in
                       ("id", "pdf", "title", "authors", "institutions")}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote chosen to {chosen_path}")


if __name__ == "__main__":
    main()
