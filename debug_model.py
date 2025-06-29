#!/usr/bin/env python3
"""
Debug script for testing OpenAI model component.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Debug OpenAI model."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return 1
    
    try:
        # Test OpenAI model
        logger.info("Testing OpenAI model...")
        
        # Initialize model
        model = OpenAIModel(
            model_name="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,
            max_tokens=10
        )
        
        # Test simple generation
        logger.info("Testing text generation...")
        prompt = "What is 2 + 2? Answer with just the number."
        response = model.generate(prompt)
        logger.info(f"Generation response: {response}")
        
        # Test multiple choice question
        logger.info("Testing multiple choice question...")
        question = "What is the capital of France?"
        choices = ["London", "Paris", "Berlin", "Madrid"]
        
        # Test direct prediction
        logger.info("Testing direct answer prediction...")
        predicted_letter = model.predict_answer(question, choices)
        logger.info(f"Predicted answer: {predicted_letter}")
        
        # Test scoring
        logger.info("Testing choice scoring...")
        scores = model.score(question, choices)
        logger.info(f"Choice scores: {scores}")
        
        # Test log probabilities
        logger.info("Testing log probabilities...")
        for i, choice in enumerate(choices):
            logprobs = model.logprobs(question, choice)
            logger.info(f"Logprobs for '{choice}': {logprobs}")
        
        logger.info("OpenAI model test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing OpenAI model: {e}")
        logger.exception("Full error details:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 