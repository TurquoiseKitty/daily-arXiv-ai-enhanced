import os
import json
import argparse
import pdfplumber
import torch
import warnings
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,
    pipeline
)

# 1) Silence the CropBox warnings from pdfminer/pdfplumber:
warnings.filterwarnings(
    "ignore",
    message=r"CropBox missing from /Page.*",
    module="pdfplumber"
)
# 2) (Optionally) suppress lower-level pdfminer logs entirely:
logging.getLogger("pdfminer").setLevel(logging.ERROR)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract author institutions from PDFs using a quantized Qwen-7B LLM"
    )
    parser.add_argument(
        # "--papers_dir", required=True,
        "papers_dir",
        help="Directory containing PDF files (e.g. XXX_pdf)"
    )
    return parser.parse_args()


def load_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # 4-bit quantization for fitting within ~4-8GB GPU VRAM
    '''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="D:/CODING/huggingface",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        cache_dir="D:/CODING/huggingface",
        device_map="auto",
        trust_remote_code=True
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=512,
        # temperature=0.0,
        do_sample=False,
        # top_k=0,           # disable top-k sampling
        # top_p=1.0,         # disable nucleus sampling
        return_full_text=False,
    )


def extract_text(path: str, max_pages: int = 1) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:max_pages]:
            txt = page.extract_text() or ""
            texts.append(txt)
    return "\n".join(texts)


def extract_affiliations(gen, text: str) -> list:
    prompt = (
        "You are an assistant that extracts author institutions from scientific papers. \n\n"
        "Given a text snippet, extract **only** a JSON array of unique institution names from the snippet below.\n\n"
        "Snippet:\n"
        f"{text[:2000]}\n\n"
        "Output:\n\n"
    )
    output = gen(prompt)[0]["generated_text"]
    try:
        start = output.index('[')
        end = output.rindex(']')
        data = json.loads(output[start:end+1])
    except Exception:
        lines = output.splitlines()
        data = [ln.strip('-• ') for ln in lines if ln.strip()]
    return list(dict.fromkeys(data))


def main():
    args = parse_args()
    # Derive prefix from folder name
    folder = os.path.basename(os.path.normpath(args.papers_dir))
    prefix = folder[:-4] if folder.endswith("_pdf") else folder
    output_path = f"{prefix}_affiliation.jsonl"

    # Ensure HF cache configured before loading model (if needed)
    # os.environ['HF_HOME'] = "/path/to/huggingface/cache"

    print(f"Loading model for inference...")
    generator = load_model()

    with open(output_path, 'w', encoding='utf-8') as out_file:
        for fname in sorted(os.listdir(args.papers_dir)):
            if not fname.lower().endswith('.pdf'):
                continue
            pdf_path = os.path.join(args.papers_dir, fname)
            print(f"Processing {fname}...")
            text = extract_text(pdf_path)
            if not text:
                print(f"  Warning: no text extracted from {fname}")
                continue
            institutions = extract_affiliations(generator, text)
            record = {"paper": fname, "institutions": institutions}
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Affiliations saved to {output_path}")


if __name__ == '__main__':
    # os.environ['HF_HOME'] = "D:/CODING/huggingface"
    main()
