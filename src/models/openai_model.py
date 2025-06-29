"""
OpenAI model implementation for the evaluation framework.
Supports GPT-4o-mini and other OpenAI models.
"""

import os
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import logging

from models.base import BaseModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI model.
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o-mini')
            api_key: OpenAI API key (if not provided, will use environment variable)
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Set up API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Default generation parameters
        self.default_params = {
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        self.default_params.update(kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        params = self.default_params.copy()
        params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with {self.model_name}: {e}")
            raise
    
    def logprobs(self, prompt: str, completion: str) -> List[float]:
        """
        Get log probabilities for a completion using OpenAI API.
        
        Args:
            prompt: Input prompt
            completion: Text to get log probabilities for
            
        Returns:
            List of log probabilities for each token in completion
        """
        try:
            # For OpenAI models, we need to use the completion API with logprobs
            full_text = prompt + completion
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_text}],
                logprobs=True,
                top_logprobs=1,
                max_tokens=1  # We only need 1 token to get logprobs for the completion
            )
            
            # Extract log probabilities for the completion part
            logprobs = []
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for item in response.choices[0].logprobs.content:
                    if item.logprob is not None:
                        logprobs.append(item.logprob)
            
            return logprobs
        except Exception as e:
            logger.error(f"Error getting logprobs with {self.model_name}: {e}")
            # Return a default low score if logprobs fail
            return [float('-inf')]
    
    def score(self, prompt: str, choices: List[str]) -> List[float]:
        """
        Score multiple choice options using log probabilities.
        
        Args:
            prompt: Input prompt (question)
            choices: List of possible answers
            
        Returns:
            List of scores for each choice
        """
        scores = []
        
        for choice in choices:
            try:
                # Get log probabilities for this choice
                logprobs = self.logprobs(prompt, choice)
                # Sum log probabilities for the entire choice
                score = sum(logprobs) if logprobs else float('-inf')
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error scoring choice '{choice}': {e}")
                scores.append(float('-inf'))
        
        return scores
    
    def predict_answer(self, prompt: str, choices: List[str]) -> str:
        """
        Predict the answer by asking the model to choose A, B, C, or D.
        
        Args:
            prompt: Input prompt (question)
            choices: List of possible answers
            
        Returns:
            Predicted answer (A, B, C, or D)
        """
        try:
            # Format the prompt to ask for A, B, C, or D
            formatted_prompt = f"{prompt}\n\nPlease respond with only the letter (A, B, C, or D) of the correct answer."
            
            response = self.generate(formatted_prompt, max_tokens=5)
            
            # Extract the letter from the response
            response = response.strip().upper()
            if response in ['A', 'B', 'C', 'D']:
                return response
            else:
                # If the response doesn't contain a valid letter, try to extract it
                for char in response:
                    if char in ['A', 'B', 'C', 'D']:
                        return char
                
                # Default to A if no valid letter found
                return 'A'
                
        except Exception as e:
            logger.warning(f"Error predicting answer: {e}")
            return 'A'  # Default fallback 