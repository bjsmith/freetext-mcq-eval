#!/usr/bin/env python3
"""
Debug script for testing the full evaluation pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.openai_model import OpenAIModel
from benchmarks.mmlu import MMLUBenchmark
from graders.logprob_grader import LogProbGrader
from evaluator import Evaluator

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for maximum detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Debug the full evaluation pipeline."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return 1
    
    try:
        logger.info("=== STARTING EVALUATION PIPELINE DEBUG ===")
        
        # Step 1: Initialize model
        logger.info("Step 1: Initializing OpenAI model...")
        model = OpenAIModel(
            model_name="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,
            max_tokens=10
        )
        logger.info(f"Model initialized: {model}")
        
        # Step 2: Initialize benchmark
        logger.info("Step 2: Initializing MMLU benchmark...")
        test_subjects = ['computer_security']  # Just one subject for debugging
        benchmark = MMLUBenchmark(subjects=test_subjects)
        logger.info(f"Benchmark initialized: {benchmark}")
        
        # Step 3: Initialize grader
        logger.info("Step 3: Initializing log probability grader...")
        grader = LogProbGrader()
        logger.info(f"Grader initialized: {grader}")
        
        # Step 4: Load benchmark data
        logger.info("Step 4: Loading benchmark data...")
        benchmark.load_data()
        logger.info("Benchmark data loaded successfully")
        
        # Step 5: Get questions
        logger.info("Step 5: Getting questions...")
        questions = list(benchmark.get_questions())
        logger.info(f"Retrieved {len(questions)} questions")
        
        # Step 6: Test with just 2 questions
        logger.info("Step 6: Testing with 2 questions...")
        test_questions = questions[:2]
        
        for i, question_data in enumerate(test_questions):
            logger.info(f"\n--- Processing Question {i+1} ---")
            
            # Format question
            formatted_question = benchmark.format_question(question_data)
            choices = benchmark.get_choices(question_data)
            correct_answer = benchmark.get_correct_answer(question_data)
            
            logger.info(f"Question: {question_data['question'][:100]}...")
            logger.info(f"Choices: {choices}")
            logger.info(f"Correct Answer: {correct_answer} ({choices[correct_answer]})")
            logger.info(f"Formatted Question: {formatted_question[:200]}...")
            
            # Test direct prediction
            logger.info("Testing direct prediction...")
            predicted_letter = model.predict_answer(formatted_question, choices)
            predicted_answer = ord(predicted_letter) - ord('A')
            logger.info(f"Predicted: {predicted_letter} ({choices[predicted_answer]})")
            logger.info(f"Correct: {predicted_answer == correct_answer}")
            
            # Test scoring
            logger.info("Testing log probability scoring...")
            scores = model.score(formatted_question, choices)
            logger.info(f"Scores: {scores}")
            
            # Test grading
            logger.info("Testing grading...")
            grade_result = grader.grade_question(scores, correct_answer)
            logger.info(f"Grade result: {grade_result}")
        
        # Step 7: Test full evaluator
        logger.info("Step 7: Testing full evaluator...")
        evaluator = Evaluator(model, benchmark, grader)
        
        # Test direct method
        logger.info("Testing direct method...")
        summary_direct = evaluator.evaluate(
            max_questions=2,
            batch_size=1,
            method="direct"
        )
        logger.info(f"Direct method summary: {summary_direct}")
        
        # Test logprob method
        logger.info("Testing logprob method...")
        evaluator = Evaluator(model, benchmark, grader)  # Reset evaluator
        summary_logprob = evaluator.evaluate(
            max_questions=2,
            batch_size=1,
            method="logprob"
        )
        logger.info(f"Logprob method summary: {summary_logprob}")
        
        logger.info("=== EVALUATION PIPELINE DEBUG COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        logger.exception("Full error details:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 