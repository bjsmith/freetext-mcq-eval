#!/usr/bin/env python3
"""
Example script to test GPT-4o-mini with MMLU benchmark.
This demonstrates how to use the evaluation framework.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.openai_model import OpenAIModel
from benchmarks.mmlu import MMLUBenchmark
from graders.logprob_grader import LogProbGrader
from evaluator import Evaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to run the evaluation."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.info("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize model
        logger.info("Initializing GPT-4o-mini model...")
        model = OpenAIModel(
            model_name="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,  # Deterministic responses
            max_tokens=10     # Short responses for multiple choice
        )
        
        # Initialize benchmark (start with a few subjects for testing)
        logger.info("Initializing MMLU benchmark...")
        test_subjects = [
            'high_school_mathematics',
            'high_school_physics',
            'computer_security'
        ]
        benchmark = MMLUBenchmark(subjects=test_subjects)
        
        # Initialize grader
        logger.info("Initializing log probability grader...")
        grader = LogProbGrader()
        
        # Run evaluation with limited questions for testing
        logger.info("Starting evaluation...")
        
        # Test both methods
        methods = ["direct", "logprob"]
        
        for method in methods:
            logger.info(f"\nTesting {method.upper()} method:")
            evaluator = Evaluator(model, benchmark, grader)
            
            summary = evaluator.evaluate(
                max_questions=10,  # Start with 10 questions for testing
                batch_size=1,      # Process one question at a time
                method=method
            )
            
            # Display results
            print(f"\n{method.upper()} METHOD RESULTS:")
            print("="*50)
            print(f"Model: {summary['model']}")
            print(f"Benchmark: {summary['benchmark']}")
            print(f"Method: {method}")
            print(f"Total Questions: {summary['total_questions']}")
            print(f"Correct Answers: {summary['correct_answers']}")
            print(f"Accuracy: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
            print(f"Evaluation Time: {summary['evaluation_time']:.2f} seconds")
            print(f"Questions/Second: {summary['questions_per_second']:.2f}")
            
            if method == "logprob":
                print(f"Average Confidence: {summary['avg_confidence']:.4f}")
                print(f"Average Predicted Probability: {summary['avg_predicted_probability']:.4f}")
                print(f"Average Correct Probability: {summary['avg_correct_probability']:.4f}")
            
            # Subject-wise results
            if summary['subject_accuracy']:
                print("\nSubject-wise Accuracy:")
                for subject, acc in sorted(summary['subject_accuracy'].items()):
                    stats = summary['subject_stats'][subject]
                    print(f"  {subject}: {acc:.4f} ({acc*100:.2f}%) - {stats['correct']}/{stats['total']} correct")
            
            # Save results
            output_dir = Path(f"results_{method}")
            evaluator.save_results(output_dir)
            print(f"\nResults saved to: {output_dir.absolute()}")
            
            # Show some example results
            results = evaluator.get_results()
            if results:
                print(f"\nExample {method.upper()} Results:")
                for i, result in enumerate(results[:2]):  # Show first 2 results
                    print(f"\nQuestion {i+1}:")
                    print(f"  Question: {result['question'][:100]}...")
                    print(f"  Choices: {result['choices']}")
                    print(f"  Correct Answer: {result['correct_answer']} ({result['choices'][result['correct_answer']]})")
                    if method == "direct":
                        print(f"  Predicted Answer: {result['predicted_letter']} ({result['choices'][result['predicted_answer']]})")
                    else:
                        print(f"  Predicted Answer: {result['predicted_answer']} ({result['choices'][result['predicted_answer']]})")
                    print(f"  Correct: {result['correct']}")
                    print(f"  Subject: {result['subject']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full error details:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 