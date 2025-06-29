"""
Base grader interface for the evaluation framework.
All grading algorithms should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseGrader(ABC):
    """Base class for all grading algorithms."""
    
    def __init__(self, grader_name: str, **kwargs):
        """
        Initialize the grader.
        
        Args:
            grader_name: Name/identifier for the grader
            **kwargs: Additional grader-specific parameters
        """
        self.grader_name = grader_name
        self.config = kwargs
        logger.info(f"Initialized grader: {grader_name}")
    
    @abstractmethod
    def grade_question(self, model_scores: List[float], correct_answer: int) -> Dict[str, Any]:
        """
        Grade a single question based on model scores.
        
        Args:
            model_scores: List of scores for each choice
            correct_answer: Index of the correct answer
            
        Returns:
            Dictionary containing grading results with keys like:
            - 'correct': Boolean indicating if correct answer was chosen
            - 'predicted_answer': Index of predicted answer
            - 'confidence': Confidence score
            - 'scores': Original scores
            - 'metadata': Additional metadata
        """
        pass
    
    @abstractmethod
    def grade_batch(self, model_scores_batch: List[List[float]], correct_answers: List[int]) -> List[Dict[str, Any]]:
        """
        Grade a batch of questions.
        
        Args:
            model_scores_batch: List of score lists for each question
            correct_answers: List of correct answer indices
            
        Returns:
            List of grading results for each question
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.grader_name})"
    
    def __repr__(self) -> str:
        return self.__str__() 