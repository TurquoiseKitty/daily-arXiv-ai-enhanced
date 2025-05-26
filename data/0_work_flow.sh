#!/usr/bin/env bash
set -euo pipefail

# Base name (no spaces around =)
date="2025-05-25"

# Download PDFs
python downloader.py "${date}.jsonl"

# Extract affiliations & mark preferences
python affiliation_extract.py "${date}_pdf" "${date}.jsonl"

# Remove the PDF folder when done
if [ -d "${date}_pdf" ]; then
  rm -rf "${date}_pdf"
fi