"""
Base benchmark interface for the evaluation framework.
All benchmark implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for all benchmark implementations."""
    
    def __init__(self, benchmark_name: str, **kwargs):
        """
        Initialize the benchmark.
        
        Args:
            benchmark_name: Name/identifier for the benchmark
            **kwargs: Additional benchmark-specific parameters
        """
        self.benchmark_name = benchmark_name
        self.config = kwargs
        logger.info(f"Initialized benchmark: {benchmark_name}")
    
    @abstractmethod
    def load_data(self) -> None:
        """
        Load the benchmark data.
        This method should be called before using the benchmark.
        """
        pass
    
    @abstractmethod
    def get_questions(self) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator over all questions in the benchmark.
        
        Yields:
            Dictionary containing question data with keys like:
            - 'question': The question text
            - 'choices': List of possible answers
            - 'correct_answer': Index of correct answer
            - 'subject': Subject/category (if applicable)
            - 'metadata': Additional metadata
        """
        pass
    
    @abstractmethod
    def format_question(self, question_data: Dict[str, Any]) -> str:
        """
        Format a question for model input.
        
        Args:
            question_data: Question data from get_questions()
            
        Returns:
            Formatted question string
        """
        pass
    
    @abstractmethod
    def get_choices(self, question_data: Dict[str, Any]) -> List[str]:
        """
        Get the choices for a question.
        
        Args:
            question_data: Question data from get_questions()
            
        Returns:
            List of choice strings
        """
        pass
    
    @abstractmethod
    def get_correct_answer(self, question_data: Dict[str, Any]) -> int:
        """
        Get the correct answer index for a question.
        
        Args:
            question_data: Question data from get_questions()
            
        Returns:
            Index of correct answer
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.benchmark_name})"
    
    def __repr__(self) -> str:
        return self.__str__() 