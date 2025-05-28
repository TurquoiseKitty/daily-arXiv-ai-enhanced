#!/usr/bin/env bash
set -euo pipefail

date="2025-05-27"

# 1) Download PDFs
python downloader.py "${date}.jsonl"

# 2) Extract affiliations & mark preferences
python affiliation_extract.py "${date}_pdf" "${date}.jsonl"

# 3) Keep only the selected PDFs (delete the rest)
python filter_pdfs.py "${date}_pdf" "${date}_chosen_affiliation.jsonl"

echo "Done. Only selected PDFs remain in ${date}_pdf."
