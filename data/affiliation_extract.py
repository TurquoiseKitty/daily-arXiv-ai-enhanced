#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re
import pdfplumber
import torch
import warnings
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1) Silence the CropBox warnings from pdfminer/pdfplumber:
warnings.filterwarnings(
    "ignore",
    message=r"CropBox missing from /Page.*",
    module="pdfplumber"
)
# 2) Suppress lower-level pdfminer logs entirely:
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge affiliations and select preferred-company papers via regex matching"
    )
    parser.add_argument(
        "papers_dir",
        help="Directory containing PDF files (e.g. XXX_pdf)"
    )
    parser.add_argument(
        "existing_jsonl",
        help="Path to your existing JSONL file"
    )
    return parser.parse_args()


def load_model():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        temperature=0.6,
        max_new_tokens=2048,
        return_full_text=False,
    )


def extract_text(path: str, max_pages: int = 2, footer_margin: float = 50) -> str:
    """
    Extract text from the first `max_pages` pages and add footer words from page 1.
    """
    pages_text = []
    footer_text = []
    with pdfplumber.open(path) as pdf:
        # extract first N pages
        for page in pdf.pages[:max_pages]:
            pages_text.append(page.extract_text() or "")
        # extract footer from first page
        first = pdf.pages[0]
        height = first.height
        words = first.extract_words()
        for w in words:
            if float(w['bottom']) >= height - footer_margin:
                footer_text.append(w['text'])
    combined = "\n".join(pages_text)
    if footer_text:
        combined += "\n" + " ".join(footer_text)
    return combined


def extract_affiliations(gen, text: str) -> list:
    think_prompt = (
        "You are an assistant that extracts author institutions from scientific papers.\n"
        "First, think step by step about which institutions appear in the snippet below, and then "
        "at the end clearly list your conclusions.\n\n"
        "Snippet:\n"
        f"{text[:2000]}\n\n"
        "Your reasoning and conclusion:\n"
    )
    reasoning = gen(think_prompt)[0]["generated_text"]
    extract_prompt = (
        "Based on the reasoning and conclusion above, output **only** a JSON array of unique institution names.\n\n"
        f"{reasoning}\n\n"
        "Output:\n"
    )
    output = gen(extract_prompt)[0]["generated_text"]
    try:
        start = output.index('[')
        end = output.rindex(']')
        data = json.loads(output[start:end+1])
    except Exception:
        lines = output.splitlines()
        data = [ln.strip('-• ') for ln in lines if ln.strip()]
    return list(dict.fromkeys(data))

# Company regex patterns
companies = [
    r"open\s*ai",
    r"meta",
    r"google",
    r"deep\s*mind",
    r"anthropic",
    r"microsoft",
    r"byte\s*dance"
]
pattern = re.compile(r"\b(?:" + "|".join(companies) + r")\b", re.IGNORECASE)

def matches_company(institutions: list) -> bool:
    return any(pattern.search(inst) for inst in institutions)


def main():
    args = parse_args()
    # derive output file names
    folder = os.path.basename(os.path.normpath(args.papers_dir))
    prefix = folder[:-4] if folder.endswith("_pdf") else folder
    merged_path = f"{prefix}_affiliation.jsonl"
    chosen_path = f"{prefix}_chosen_affiliation.jsonl"

    # load existing JSONL
    existing = {}
        # load existing JSONL, skipping blank/malformed lines
    existing = {}
    with open(args.existing_jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line: {line[:30]}...")
                continue
            rid = rec.get("id")
            if rid and rid not in existing:
                existing[rid] = rec

    # load LLM
    print("Loading model...")
    gen = load_model()

    # process PDFs
    for fname in sorted(os.listdir(args.papers_dir)):
        if not fname.lower().endswith('.pdf'):
            continue
        pid = os.path.splitext(fname)[0]
        path = os.path.join(args.papers_dir, fname)
        print(f"Processing {fname}...")
        txt = extract_text(path)
        if not txt.strip():
            print(f"  Warning: no text from {fname}")
            continue

        # initial extraction
        insts = extract_affiliations(gen, txt)

        # fallback layout-aware extraction if empty
        if not insts:
            print("  No affiliations found—trying layout-aware fallback on page 1...")
            with pdfplumber.open(path) as pdf:
                page1 = pdf.pages[0]
                layout_txt = page1.extract_text(layout=True) or ""
            insts = extract_affiliations(gen, layout_txt)

        if pid in existing:
            rec = existing[pid]
            rec['institutions'] = list(dict.fromkeys(rec.get('institutions', []) + insts))
            rec['preference'] = 1 if matches_company(rec['institutions']) else 0
        else:
            print(f"  Warning: {pid} not in existing JSONL")

    # write merged
    with open(merged_path, 'w', encoding='utf-8') as fout:
        for rec in existing.values():
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote merged to {merged_path}")

    # write chosen
    with open(chosen_path, 'w', encoding='utf-8') as fout:
        for rec in existing.values():
            if rec.get('preference') == 1:
                out = {k: rec.get(k) for k in ['id','pdf','title','authors','institutions']}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote chosen to {chosen_path}")

if __name__ == '__main__':
    main()
