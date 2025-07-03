#!/usr/bin/env python3
"""
Demo: Run MMLU evaluation using lm-eval-harness (not the original MMLU repo).
This script works with any model supported by lm-eval-harness, including HuggingFace and OpenAI models.

Requirements:
    pip install lm-eval openai

Usage:
    - Set MODEL_NAME and PROVIDER as needed.
    - Set OPENAI_API_KEY in your environment for OpenAI models.
"""

import os
from lm_eval import evaluator
#from lm_eval.models import get_model
from lm_eval.tasks import get_task_dict
#from customOpenAICompletionsAPI import OpenAICompletionsAPI
from lm_eval.models.openai_completions import OpenAICompletionsAPI


# --- CONFIG ---
PROVIDER = "openai"  # "openai" or "hf"
MODEL_NAME = "gpt-3.5-turbo"  # e.g., "gpt-3.5-turbo" for OpenAI, "EleutherAI/gpt-neo-125M" for HF
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#TASKS = ["mmlu_history", "mmlu_geography"]  # or just ["mmlu"] for all MMLU subjects
TASKS = ["mmlu_anatomy"]  # or just ["mmlu"] for all MMLU subjects

def setup_openai_model(api_key: str = None) -> OpenAICompletionsAPI:
    """Setup the OpenAI model for evaluation."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass --api-key")
    model_config = {
        "model": "gpt-4o-mini",
        "batch_size": 1,
        "max_batch_size": 1#,

        # "logprobs": True,
        # "top_logprobs": 3
    }
    return OpenAICompletionsAPI(**model_config)

def main():
    if PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
        model = setup_openai_model()
    elif PROVIDER == "hf":
        raise NotImplementedError()
        #model = get_model("hf-causal", pretrained=MODEL_NAME, device="cpu")
    else:
        raise ValueError("PROVIDER must be 'openai' or 'hf'.")

    task_dict = get_task_dict(TASKS)
    results = evaluator.evaluate(
        lm=model,
        task_dict=task_dict,
        #num_fewshot=5,
        limit=20,  # Set to a small int for a quick test
        bootstrap_iters=100#,
    )

    print("\n=== MMLU Results ===")
    for task, res in results["results"].items():
        print(f"{task}: acc={res.get('acc', 'N/A'):.4f}")

if __name__ == "__main__":
    main() 