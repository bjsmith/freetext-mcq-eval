#!/usr/bin/env python3
"""
Simple example of using lm-harness to evaluate GPT-4o on MMLU tasks.
"""

import os
from dotenv import load_dotenv
from lm_eval import evaluator
from lm_eval.models.openai_completions import OpenAICompletionsLM
from lm_eval.tasks import get_task_dict

# Load environment variables
load_dotenv()

def main():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize the model - note: api_key is NOT passed to constructor
    model = OpenAICompletionsLM(
        model="gpt-4o",
        temperature=0.0,
        max_new_tokens=512,
        batch_size=1
    )
    
    # Get tasks
    tasks = ["mmlu_history", "mmlu_geography"]
    task_dict = get_task_dict(tasks)
    
    print(f"Evaluating GPT-4o on: {', '.join(tasks)}")
    
    # Run evaluation
    results = evaluator.evaluate(
        lm=model,
        task_dict=task_dict,
        limit=None,  # All examples
        num_fewshot=0,  # No few-shot
        bootstrap_iters=1000
    )
    
    # Print results
    print("\nResults:")
    print("=" * 40)
    for task_name, task_results in results['results'].items():
        acc = task_results.get('acc', 'N/A')
        print(f"{task_name}: {acc:.4f}")

if __name__ == "__main__":
    main() 