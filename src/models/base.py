"""
Base model interface for the evaluation framework.
All model implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all model implementations."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_name: Name/identifier for the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs
        logger.info(f"Initialized model: {model_name}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion for a given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (max_tokens, temperature, etc.)
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def logprobs(self, prompt: str, completion: str) -> List[float]:
        """
        Get log probabilities for a given completion.
        
        Args:
            prompt: Input prompt
            completion: Text to get log probabilities for
            
        Returns:
            List of log probabilities for each token in completion
        """
        pass
    
    @abstractmethod
    def score(self, prompt: str, choices: List[str]) -> List[float]:
        """
        Score multiple choice options for a given prompt.
        
        Args:
            prompt: Input prompt (question)
            choices: List of possible answers
            
        Returns:
            List of scores for each choice
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"
    
    def __repr__(self) -> str:
        return self.__str__() 