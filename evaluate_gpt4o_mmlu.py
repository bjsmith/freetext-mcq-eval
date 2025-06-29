#!/usr/bin/env python3
"""
Script to evaluate GPT-4o on MMLU history/geography tasks using lm-harness.
"""

import os
import json
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv
from lm_eval import evaluator
from lm_eval.models.openai_completions import OpenAICompletionsLM
from lm_eval.tasks import get_task_dict

# Load environment variables
load_dotenv()

def setup_openai_model(api_key: str = None) -> OpenAICompletionsLM:
    """Setup the OpenAI model for evaluation."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass --api-key")
    
    # Configure the model
    model_config = {
        "model": "gpt-4o",
        "batch_size": 1,  # Adjust based on your needs
        "max_batch_size": 1,
        "device": "cpu",  # Not used for OpenAI models but required
        "trust_remote_code": False,
        "use_fast_tokenizer": False,
        "revision": None,
        "use_auth_token": None,
        "low_cpu_mem_usage": False,
        "torch_dtype": None,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "bfloat16": False,
        "fp16": False,
        "model_kwargs": {},
        "peft": None,
        "tokenizer": None,
        "tokenizer_kwargs": {},
        "add_bos_token": False,
        "add_eos_token": False,
        "prefix_token_id": None,
        "chat_template": None,
        "truncation": False,
        "max_length": None,
        "truncation_side": "right",
        "padding": False,
        "add_special_tokens": False,
        "stopping_criteria": None,
        "do_sample": False,
        "use_cache": True,
        "num_return_sequences": 1,
        "temperature": 0.0,  # Use 0 temperature for deterministic evaluation
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_repetition_penalty": 1.0,
        "bad_words_ids": None,
        "min_length": 0,
        "max_new_tokens": 512,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "output_scores": False,
        "return_dict_in_generate": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_decoder_ids": None,
        "sequence_bias": None,
        "guidance_scale": None,
        "logits_processor": None,
        "scheduler": None,
        "function_to_apply": None,
        "raw_scores": False,
        "content_filter": False,
        "seed": None,
        "logprobs": None,
        "echo": False,
        "suffix": None,
        "best_of": None,
        "logit_bias": None,
        "user": None,
        "api_base": None,
        "api_version": None,
        "organization": None,
        "api_key": api_key,
        "parallel_tool_calls": False,
        "max_retries": 3,
        "retry_delay": 1.0,
        "timeout": 60.0,
    }
    
    return OpenAICompletionsLM(**model_config)

def get_mmlu_tasks() -> List[str]:
    """Get the MMLU history and geography tasks."""
    return [
        "mmlu_history",
        "mmlu_geography"
    ]

def run_evaluation(model: OpenAICompletionsLM, tasks: List[str], output_file: str = None) -> Dict[str, Any]:
    """Run the evaluation on specified tasks."""
    print(f"Evaluating GPT-4o on tasks: {', '.join(tasks)}")
    
    # Get task dictionary
    task_dict = get_task_dict(tasks)
    
    # Run evaluation
    results = evaluator.evaluate(
        lm=model,
        task_dict=task_dict,
        limit=None,  # Evaluate all examples
        write_out=False,
        log_samples=False,
        description_dict=None,
        check_integrity=False,
        decontamination_ngrams_path=None,
        output_path=None,
        random_seed=42,
        num_fewshot=0,  # No few-shot examples for MMLU
        bootstrap_iters=1000,
    )
    
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
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on MMLU history/geography tasks")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", type=str, default="gpt4o_mmlu_results.json", 
                       help="Output file for results (default: gpt4o_mmlu_results.json)")
    parser.add_argument("--tasks", nargs="+", choices=["mmlu_history", "mmlu_geography"], 
                       default=["mmlu_history", "mmlu_geography"],
                       help="Tasks to evaluate (default: both history and geography)")
    
    args = parser.parse_args()
    
    try:
        # Setup model
        print("Setting up GPT-4o model...")
        model = setup_openai_model(args.api_key)
        
        # Run evaluation
        results = run_evaluation(model, args.tasks, args.output)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 