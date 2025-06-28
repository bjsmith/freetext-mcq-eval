"""
Model interface for different LLM models to evaluate on MMLU.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional, Union
import numpy as np
from tqdm import tqdm
import time


class BaseModelInterface:
    """Base class for model interfaces."""
    
    def __init__(self, model_name: str):
        """
        Initialize the model interface.
        
        Args:
            model_name: Name or path of the model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer. To be implemented by subclasses."""
        raise NotImplementedError
        
    def generate_answer(self, question: str) -> str:
        """
        Generate an answer for a given question.
        
        Args:
            question: Formatted question string
            
        Returns:
            Model's answer
        """
        raise NotImplementedError
        
    def generate_answers_batch(self, questions: List[str]) -> List[str]:
        """
        Generate answers for a batch of questions.
        
        Args:
            questions: List of formatted question strings
            
        Returns:
            List of model answers
        """
        answers = []
        for question in tqdm(questions, desc="Generating answers"):
            answer = self.generate_answer(question)
            answers.append(answer)
        return answers


class HuggingFaceModelInterface(BaseModelInterface):
    """Interface for Hugging Face models."""
    
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        """
        Initialize the Hugging Face model interface.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run the model on
            max_length: Maximum length for generation
        """
        super().__init__(model_name)
        self.device = device
        self.max_length = max_length
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        # Determine device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully!")
        
    def generate_answer(self, question: str) -> str:
        """
        Generate an answer for a given question.
        
        Args:
            question: Formatted question string
            
        Returns:
            Model's answer
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Tokenize input
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, 
                               max_length=self.max_length)
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new tokens (after the input)
        input_length = inputs['input_ids'].shape[1]
        response = response[input_length:].strip()
        
        return response


class PipelineModelInterface(BaseModelInterface):
    """Interface for Hugging Face pipeline models."""
    
    def __init__(self, model_name: str, task: str = "text-generation", device: int = -1):
        """
        Initialize the pipeline model interface.
        
        Args:
            model_name: Hugging Face model name
            task: Pipeline task type
            device: Device to run the model on (-1 for CPU, 0 for GPU)
        """
        super().__init__(model_name)
        self.task = task
        self.device = device
        self.pipeline = None
        
    def load_model(self):
        """Load the model pipeline."""
        print(f"Loading pipeline model: {self.model_name}")
        
        self.pipeline = pipeline(
            self.task,
            model=self.model_name,
            device=self.device,
            torch_dtype=torch.float16 if self.device >= 0 else torch.float32
        )
        
        print("Pipeline model loaded successfully!")
        
    def generate_answer(self, question: str) -> str:
        """
        Generate an answer for a given question.
        
        Args:
            question: Formatted question string
            
        Returns:
            Model's answer
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_model() first.")
            
        # Generate response
        response = self.pipeline(
            question,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the generated text
        if isinstance(response, list):
            generated_text = response[0]['generated_text']
        else:
            generated_text = response['generated_text']
            
        # Remove the input question from the response
        answer = generated_text[len(question):].strip()
        
        return answer


class MockModelInterface(BaseModelInterface):
    """Mock model interface for testing purposes."""
    
    def __init__(self, accuracy: float = 0.75):
        """
        Initialize the mock model interface.
        
        Args:
            accuracy: Simulated accuracy of the mock model
        """
        super().__init__("mock_model")
        self.accuracy = accuracy
        
    def load_model(self):
        """Mock model loading."""
        print("Loading mock model...")
        print("Mock model loaded successfully!")
        
    def generate_answer(self, question: str) -> str:
        """
        Generate a mock answer.
        
        Args:
            question: Formatted question string
            
        Returns:
            Mock answer (A, B, C, or D)
        """
        # Simulate some processing time
        time.sleep(0.1)
        
        # Randomly generate correct or incorrect answer
        if np.random.random() < self.accuracy:
            # Generate correct answer (extract from question)
            if "Answer:" in question:
                # Extract the correct answer from the question
                lines = question.split('\n')
                for line in lines:
                    if line.strip().startswith('Answer:'):
                        return line.strip().split(':')[-1].strip()
            
            # If we can't extract, return random correct answer
            return np.random.choice(['A', 'B', 'C', 'D'])
        else:
            # Return random incorrect answer
            return np.random.choice(['A', 'B', 'C', 'D'])


def create_model_interface(model_name: str, model_type: str = "huggingface", **kwargs) -> BaseModelInterface:
    """
    Factory function to create model interfaces.
    
    Args:
        model_name: Name of the model
        model_type: Type of model interface ('huggingface', 'pipeline', 'mock')
        **kwargs: Additional arguments for the model interface
        
    Returns:
        Model interface instance
    """
    if model_type == "huggingface":
        return HuggingFaceModelInterface(model_name, **kwargs)
    elif model_type == "pipeline":
        return PipelineModelInterface(model_name, **kwargs)
    elif model_type == "mock":
        return MockModelInterface(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test the model interface."""
    # Test with mock model
    mock_model = create_model_interface("mock_model", model_type="mock", accuracy=0.8)
    mock_model.load_model()
    
    # Test question
    test_question = """Question: What is 2 + 2?

Options:
A. 3
B. 4
C. 5
D. 6

Answer:"""
    
    answer = mock_model.generate_answer(test_question)
    print(f"Mock model answer: {answer}")
    
    # Test batch generation
    questions = [test_question] * 3
    answers = mock_model.generate_answers_batch(questions)
    print(f"Batch answers: {answers}")


if __name__ == "__main__":
    main() 