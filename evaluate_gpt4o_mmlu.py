#!/usr/bin/env python3
"""
Script to evaluate GPT-4o on MMLU history/geography tasks using lm-harness.
"""

import os
import json
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv
# from lm_eval.models.openai_completions import OpenAICompletionsAPI
from customOpenAICompletionsAPI import OpenAICompletionsAPI
from lm_eval import evaluator
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager

# Load environment variables
load_dotenv()

def setup_openai_model(api_key: str = None) -> OpenAICompletionsAPI:
    """Setup the OpenAI model for evaluation."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass --api-key")
    model_config = {
        "model": "gpt-4o-mini",
        "batch_size": 1,
        "max_batch_size": 1,
    }
    return OpenAICompletionsAPI(**model_config)

def get_mmlu_tasks(mode: str) -> Dict[str, dict]:
    """Get the MMLU tasks with the correct grader for the selected mode."""
    # Example: use 'direct' or 'loglikelihood' grader
    grader = "direct" if mode == "direct" else "loglikelihood"
    # You can add more tasks here as needed
    tasks = {
        "mmlu_social_sciences": {
            "task": "mmlu_social_sciences",
            "grader": grader
        },
        # Add more tasks as needed
    }
    return tasks

def run_evaluation(model: OpenAICompletionsAPI, tasks: Dict[str, dict], output_file: str = None, mode: str = "direct") -> Dict[str, Any]:
    """Run the evaluation on specified tasks."""
    print(f"Evaluating GPT-4o on tasks: {', '.join(tasks.keys())} (mode: {mode})")
    # Build the task dict for lm-eval
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=5,
        batch_size=1,
        limit=100,
        bootstrap_iters=0,  # Set to 1000+ for confidence intervals
        description_dict={},
        write_out=True,
        log_samples=True
    )
    # task_dict = {}
    # for task_name, task_info in tasks.items():
    #     # Use TaskManager to get the base task config
    #     task_manager = TaskManager()
    #     base_task = task_manager.get_task(task_info["task"])
    #     # Override the grader
    #     base_task._config["grader"] = task_info["grader"]
    #     task_dict[task_name] = base_task
    # Run evaluation

        
    # results = evaluator.evaluate(
    #     lm=model,
    #     task_dict=task_dict,
    #     limit=None,
    #     write_out=False,
    #     log_samples=False,
    #     description_dict=None,
    #     check_integrity=False,
    #     decontamination_ngrams_path=None,
    #     output_path=None,
    #     random_seed=42,
    #     num_fewshot=0,
    #     bootstrap_iters=1000,
    # )
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for task_name, task_results in results['results'].items():
        print(f"\nTask: {task_name}")
        print(f"Accuracy: {task_results.get('acc', 'N/A'):.4f}")
        print(f"Accuracy (normalized): {task_results.get('acc_norm', 'N/A'):.4f}")
        print(f"Number of examples: {task_results.get('alias', 'N/A')}")
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on MMLU history/geography tasks. Use --mode to toggle between direct and logprobs (loglikelihood) grading.")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", type=str, default="gpt4o_mmlu_results.json", help="Output file for results (default: gpt4o_mmlu_results.json)")
    parser.add_argument("--mode", type=str, choices=["direct", "logprobs"], default="direct", help="Grading mode: 'direct' (default) or 'logprobs' (loglikelihood)")
    args = parser.parse_args()
    try:
        print("Setting up GPT-4o model...")
        model = setup_openai_model(args.api_key)
        #tasks = get_mmlu_tasks(args.mode)
        tasks = ["mmlu_abstract_algebra",
        "mmlu_anatomy", 
        "mmlu_astronomy",
        "mmlu_business_ethics",
        "mmlu_clinical_knowledge"]
        results = run_evaluation(model, tasks, args.output, args.mode)
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 