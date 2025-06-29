#!/usr/bin/env python3
"""
Debug script for testing MMLU benchmark component.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmarks.mmlu import MMLUBenchmark

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Debug MMLU benchmark."""
    
    try:
        # Test MMLU benchmark
        logger.info("Testing MMLU benchmark...")
        
        # Initialize benchmark with specific subjects
        test_subjects = ['computer_security', 'high_school_mathematics']
        benchmark = MMLUBenchmark(subjects=test_subjects)
        
        # Load data
        logger.info("Loading MMLU data...")
        benchmark.load_data()
        
        # Get subject stats
        stats = benchmark.get_subject_stats()
        logger.info(f"Subject stats: {stats}")
        
        # Test getting questions
        logger.info("Testing question retrieval...")
        questions = list(benchmark.get_questions())
        logger.info(f"Loaded {len(questions)} questions")
        
        # Test first few questions
        for i, question_data in enumerate(questions[:3]):
            logger.info(f"\nQuestion {i+1}:")
            logger.info(f"  Question: {question_data['question'][:100]}...")
            logger.info(f"  Choices: {question_data['choices']}")
            logger.info(f"  Correct Answer: {question_data['correct_answer']}")
            logger.info(f"  Subject: {question_data['subject']}")
            
            # Test formatting
            formatted = benchmark.format_question(question_data)
            logger.info(f"  Formatted: {formatted[:200]}...")
        
        logger.info("MMLU benchmark test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing MMLU benchmark: {e}")
        logger.exception("Full error details:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 