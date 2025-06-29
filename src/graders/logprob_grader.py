"""
Log probability grader implementation.
Uses log probabilities to score multiple choice questions.
"""

import numpy as np
from typing import List, Dict, Any
import logging

from graders.base import BaseGrader

logger = logging.getLogger(__name__)


class LogProbGrader(BaseGrader):
    """Log probability grader implementation."""
    
    def __init__(self, grader_name: str = "logprob", **kwargs):
        """
        Initialize log probability grader.
        
        Args:
            grader_name: Grader name
            **kwargs: Additional parameters
        """
        super().__init__(grader_name, **kwargs)
    
    def grade_question(self, model_scores: List[float], correct_answer: int) -> Dict[str, Any]:
        """
        Grade a single question using log probabilities.
        
        Args:
            model_scores: List of log probability scores for each choice
            correct_answer: Index of the correct answer
            
        Returns:
            Dictionary containing grading results
        """
        if not model_scores:
            raise ValueError("Model scores cannot be empty")
        
        # Convert to numpy array for easier manipulation
        scores = np.array(model_scores)
        
        # Find the predicted answer (highest score)
        predicted_answer = np.argmax(scores)
        
        # Check if correct
        correct = predicted_answer == correct_answer
        
        # Calculate confidence (difference between highest and second highest scores)
        sorted_scores = np.sort(scores)[::-1]  # Sort in descending order
        if len(sorted_scores) >= 2:
            confidence = sorted_scores[0] - sorted_scores[1]
        else:
            confidence = 0.0
        
        # Calculate softmax probabilities for additional metrics
        try:
            # Apply softmax to get probabilities
            exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
            probabilities = exp_scores / np.sum(exp_scores)
            predicted_probability = probabilities[predicted_answer]
            correct_probability = probabilities[correct_answer]
        except:
            # Fallback if softmax fails
            probabilities = None
            predicted_probability = 0.0
            correct_probability = 0.0
        
        return {
            'correct': correct,
            'predicted_answer': int(predicted_answer),
            'confidence': float(confidence),
            'scores': scores.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'predicted_probability': float(predicted_probability),
            'correct_probability': float(correct_probability),
            'metadata': {
                'grader': self.grader_name,
                'method': 'logprob'
            }
        }
    
    def grade_batch(self, model_scores_batch: List[List[float]], correct_answers: List[int]) -> List[Dict[str, Any]]:
        """
        Grade a batch of questions.
        
        Args:
            model_scores_batch: List of score lists for each question
            correct_answers: List of correct answer indices
            
        Returns:
            List of grading results for each question
        """
        if len(model_scores_batch) != len(correct_answers):
            raise ValueError("Number of score lists must match number of correct answers")
        
        results = []
        for scores, correct_answer in zip(model_scores_batch, correct_answers):
            try:
                result = self.grade_question(scores, correct_answer)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error grading question: {e}")
                # Add error result
                results.append({
                    'correct': False,
                    'predicted_answer': -1,
                    'confidence': 0.0,
                    'scores': scores,
                    'probabilities': None,
                    'predicted_probability': 0.0,
                    'correct_probability': 0.0,
                    'metadata': {
                        'grader': self.grader_name,
                        'method': 'logprob',
                        'error': str(e)
                    }
                })
        
        return results 