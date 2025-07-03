#!/usr/bin/env python3
"""
Demo: Run MMLU evaluation using the installed mmlu package (not lm-eval-harness).
This script shows how to use the official MMLU evaluation code with a HuggingFace or OpenAI model.

- For HuggingFace models: set MODEL_NAME to a valid HF model (e.g., 'EleutherAI/gpt-neo-125M')
- For OpenAI models: set USE_OPENAI=True and provide your API key

You must have the mmlu package installed and its dependencies:
    pip install mmlu  # or pip install -e . from the MMLU repo
"""

import os
import sys

# --- CONFIG ---
USE_OPENAI = False  # Set to True to use OpenAI API, False for HuggingFace
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # Or e.g. "gpt-3.5-turbo" if using OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MMLU_DATA_DIR = "data"  # Path to MMLU data (should contain 'test' and 'dev' folders)

# --- IMPORTS ---
try:
    #no version of the below is going to work. this assumes a completelyl different structure
    #in lm_harness, MMLU is only defined as a YAML file.
    from mmlu.eval import main as mmlu_main
except ImportError:
    print("[ERROR] Could not import mmlu.eval. Make sure the mmlu package is installed.")
    sys.exit(1)

# --- ARGS ---
# See eval.py --help for all options
# We'll build a minimal set of args for a quick demo

def run_mmlu_hf():
    """Run MMLU using a HuggingFace model."""
    args = [
        "--model", "hf-causal",
        "--model_args", f"pretrained={MODEL_NAME},device=cpu",
        "--tasks", "mmlu",
        "--data_dir", MMLU_DATA_DIR,
        "--output_path", "mmlu_hf_results.json",
        "--num_fewshot", "5",
        "--batch_size", "4",
        "--no_cache",
    ]
    print(f"[INFO] Running MMLU with HuggingFace model: {MODEL_NAME}")
    mmlu_main(args)

def run_mmlu_openai():
    """Run MMLU using an OpenAI model via API."""
    if not OPENAI_API_KEY:
        print("[ERROR] Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    args = [
        "--model", "openai-chat",
        "--model_args", f"model={MODEL_NAME},api_key={OPENAI_API_KEY}",
        "--tasks", "mmlu",
        "--data_dir", MMLU_DATA_DIR,
        "--output_path", "mmlu_openai_results.json",
        "--num_fewshot", "5",
        "--batch_size", "1",
        "--no_cache",
    ]
    print(f"[INFO] Running MMLU with OpenAI model: {MODEL_NAME}")
    mmlu_main(args)

if __name__ == "__main__":
    if USE_OPENAI:
        run_mmlu_openai()
    else:
        run_mmlu_hf() 