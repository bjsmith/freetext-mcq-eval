#!/usr/bin/env python3
"""
Debug version of GPT-4o MMLU evaluation with detailed logging and breakpoints.
Use this script for step-by-step debugging in VSCode.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv

# Set up detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_openai_model(api_key: str = None) -> Any:
    """Setup the OpenAI model for evaluation with debug logging."""
    logger.info("Setting up OpenAI model...")
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        logger.debug(f"API key from environment: {'SET' if api_key else 'NOT SET'}")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass --api-key")
    
    # Set the API key in environment if provided via command line
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("API key set in environment")
    
    try:
        from lm_eval.models.openai_completions import OpenAICompletionsLM
        
        # Configure the model with debug settings - note: api_key is NOT passed to constructor
        model_config = {
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "batch_size": 1,
            "timeout": 120.0,  # Longer timeout for debugging
            "max_retries": 5,  # More retries for debugging
        }
        
        logger.info(f"Model config: {model_config}")
        model = OpenAICompletionsLM(**model_config)
        logger.info("OpenAI model setup completed successfully")
        
        # BREAKPOINT: Model setup complete
        breakpoint()  # You can set breakpoints here in VSCode
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to setup OpenAI model: {e}")
        raise

def get_mmlu_tasks() -> List[str]:
    """Get the MMLU history and geography tasks with debug logging."""
    logger.info("Getting MMLU tasks...")
    tasks = ["mmlu_history", "mmlu_geography"]
    logger.info(f"Selected tasks: {tasks}")
    return tasks

def run_evaluation(model: Any, tasks: List[str], output_file: str = None) -> Dict[str, Any]:
    """Run the evaluation on specified tasks with detailed logging."""
    logger.info(f"Starting evaluation on tasks: {', '.join(tasks)}")
    
    try:
        from lm_eval import evaluator
        from lm_eval.tasks import get_task_dict
        
        # Get task dictionary
        logger.info("Loading task dictionary...")
        task_dict = get_task_dict(tasks)
        logger.info(f"Loaded {len(task_dict)} tasks")
        
        # BREAKPOINT: Before evaluation starts
        breakpoint()  # You can set breakpoints here in VSCode
        
        # Run evaluation with debug settings
        logger.info("Starting lm-eval evaluation...")
        results = evaluator.evaluate(
            lm=model,
            task_dict=task_dict,
            limit=10,  # Limit for debugging - change to None for full evaluation
            write_out=False,
            log_samples=True,  # Enable sample logging for debugging
            description_dict=None,
            check_integrity=False,
            decontamination_ngrams_path=None,
            output_path=None,
            random_seed=42,
            num_fewshot=0,
            bootstrap_iters=100,  # Reduced for debugging
        )
        
        logger.info("Evaluation completed successfully")
        
        # BREAKPOINT: After evaluation completes
        breakpoint()  # You can set breakpoints here in VSCode
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for task_name, task_results in results['results'].items():
            print(f"\nTask: {task_name}")
            print(f"Accuracy: {task_results.get('acc', 'N/A'):.4f}")
            print(f"Accuracy (normalized): {task_results.get('acc_norm', 'N/A'):.4f}")
            print(f"Number of examples: {task_results.get('alias', 'N/A')}")
            
            # Log detailed results
            logger.info(f"Task {task_name} results: {task_results}")
        
        # Save results to file if specified
        if output_file:
            logger.info(f"Saving results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def test_model_connection(model: Any) -> bool:
    """Test the model connection with a simple prompt."""
    logger.info("Testing model connection...")
    
    try:
        # Simple test prompt
        test_prompt = "What is 2+2? Answer with just the number."
        logger.info(f"Test prompt: {test_prompt}")
        
        response = model.generate(test_prompt, max_tokens=10)
        logger.info(f"Test response: {response}")
        
        print(f"Model connection test successful: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Model connection test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Debug version of GPT-4o MMLU evaluation")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", type=str, default="debug_results.json", 
                       help="Output file for results (default: debug_results.json)")
    parser.add_argument("--tasks", nargs="+", choices=["mmlu_history", "mmlu_geography"], 
                       default=["mmlu_history"],
                       help="Tasks to evaluate (default: history only for debugging)")
    parser.add_argument("--test-connection", action="store_true",
                       help="Test model connection before evaluation")
    
    args = parser.parse_args()
    
    logger.info("Starting debug evaluation script")
    logger.info(f"Arguments: {args}")
    
    try:
        # Setup model
        logger.info("Setting up GPT-4o model...")
        model = setup_openai_model(args.api_key)
        
        # Test connection if requested
        if args.test_connection:
            if not test_model_connection(model):
                logger.error("Model connection test failed. Exiting.")
                return 1
        
        # BREAKPOINT: Before running evaluation
        breakpoint()  # You can set breakpoints here in VSCode
        
        # Run evaluation
        results = run_evaluation(model, args.tasks, args.output)
        
        logger.info("Debug evaluation completed successfully!")
        print("\nDebug evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during debug evaluation: {e}")
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 