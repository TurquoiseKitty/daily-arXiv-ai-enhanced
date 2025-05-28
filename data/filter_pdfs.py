#!/usr/bin/env python3
"""
Keep only selected PDFs and remove all others from a directory.
"""

import os
import json
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Keep only selected PDFs and remove the rest"
    )
    p.add_argument("pdf_dir",        help="Directory containing PDF files")
    p.add_argument("chosen_jsonl",   help="Path to chosen affiliation JSONL file")
    return p.parse_args()

def main():
    args = parse_args()

    # Load chosen IDs
    chosen_ids = set()
    with open(args.chosen_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            if rid:
                chosen_ids.add(rid)

    # Remove any PDF whose basename isn't in chosen_ids
    for fname in os.listdir(args.pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        pid  = os.path.splitext(fname)[0]
        path = os.path.join(args.pdf_dir, fname)
        if pid not in chosen_ids:
            try:
                os.remove(path)
                print(f"Removed unselected PDF: {fname}")
            except Exception as e:
                print(f"Error removing {fname}: {e}")
        else:
            print(f"Keeping selected PDF: {fname}")

if __name__ == "__main__":
    main()
