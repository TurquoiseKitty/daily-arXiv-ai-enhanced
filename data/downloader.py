#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import logging
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIG ---
MIN_DELAY   = 1.0       # minimum delay between requests (s)
MAX_DELAY   = 3.0       # maximum delay between requests (s)
MAX_RETRIES = 5         # retry count for 429/500 errors
# ----------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def make_session():
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    })
    return session

def download_pdfs(json_path):
    # derive output dir: <basename>_pdf
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = f"{base_name}_pdf"
    os.makedirs(output_dir, exist_ok=True)
    sess = make_session()

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed line")
                continue

            paper_id = rec.get("id")
            pdf_url  = rec.get("pdf")
            if not paper_id or not pdf_url:
                logging.warning("Missing id/pdf, skipping")
                continue

            out_path = os.path.join(output_dir, f"{paper_id}.pdf")
            if os.path.exists(out_path):
                logging.info(f"[{paper_id}] already exists, skipping.")
                continue

            try:
                logging.info(f"[{paper_id}] downloading {pdf_url}")
                resp = sess.get(pdf_url, timeout=30)
                ct = resp.headers.get("Content-Type", "")
                if resp.status_code == 200 and ct.startswith("application/pdf"):
                    with open(out_path, "wb") as out:
                        out.write(resp.content)
                    logging.info(f"[{paper_id}] saved to {out_path}")
                else:
                    logging.error(
                        f"[{paper_id}] failed: HTTP {resp.status_code}, "
                        f"Content-Type={ct}"
                    )
            except requests.RequestException as e:
                logging.error(f"[{paper_id}] exception: {e}")

            # polite, randomized delay
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            logging.debug(f"Sleeping for {delay:.2f}s")
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(
        description="Download PDFs from newline-delimited arXiv JSON."
    )
    parser.add_argument(
        "json_file",
        help="Path to your newline-delimited JSON file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.json_file):
        logging.error(f"File not found: {args.json_file}")
        sys.exit(1)

    download_pdfs(args.json_file)

if __name__ == "__main__":
    main()