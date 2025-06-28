"""
Data loader for MMLU (Massive Multitask Language Understanding) dataset.
"""

from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Any, Optional
import random


class MMLUDataLoader:
    """Data loader for the MMLU dataset."""
    
    def __init__(self, dataset_name: str = "cais/mmlu"):
        """
        Initialize the MMLU data loader.
        
        Args:
            dataset_name: Name of the MMLU dataset on Hugging Face
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.subjects = None
        
    def load_dataset(self, subjects: Optional[List[str]] = None, split: str = "test"):
        """
        Load the MMLU dataset.
        
        Args:
            subjects: List of subjects to load. If None, loads all subjects.
            split: Dataset split to load ('train', 'validation', 'test')
        """
        print(f"Loading MMLU dataset: {self.dataset_name}")
        
        if subjects is None:
            # Load the 'all' config for the full dataset
            self.dataset = load_dataset(self.dataset_name, 'all', split=split)
        else:
            # Load and concatenate each subject
            all_subjects = []
            for subject in subjects:
                ds = load_dataset(self.dataset_name, subject, split=split)
                all_subjects.append(ds)
            from datasets import concatenate_datasets
            self.dataset = concatenate_datasets(all_subjects)
        
        print(f"Loaded {len(self.dataset)} examples")
        
        # Get available subjects
        if hasattr(self.dataset, 'features') and 'subject' in self.dataset.features:
            self.subjects = list(set(self.dataset['subject']))
            print(f"Available subjects: {self.subjects}")
        
        return self.dataset
    
    def get_subject_data(self, subject: str) -> pd.DataFrame:
        """
        Get data for a specific subject.
        
        Args:
            subject: Subject name
            
        Returns:
            DataFrame with subject data
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        subject_data = self.dataset.filter(lambda x: x['subject'] == subject)
        return pd.DataFrame(subject_data)
    
    def get_sample_questions(self, n: int = 5, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a sample of questions for testing.
        
        Args:
            n: Number of questions to sample
            subject: Optional subject to filter by
            
        Returns:
            List of question dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        if subject:
            filtered_data = self.dataset.filter(lambda x: x['subject'] == subject)
        else:
            filtered_data = self.dataset
            
        # Sample n questions
        sample_indices = random.sample(range(len(filtered_data)), min(n, len(filtered_data)))
        sample_questions = [filtered_data[i] for i in sample_indices]
        
        return sample_questions
    
    def format_question(self, question: Dict[str, Any]) -> str:
        """
        Format a question for model input.
        
        Args:
            question: Question dictionary from dataset
            
        Returns:
            Formatted question string
        """
        prompt = f"Question: {question['question']}\n\n"
        prompt += "Options:\n"
        choices = question['choices']
        for i, option in enumerate(['A', 'B', 'C', 'D']):
            prompt += f"{option}. {choices[i]}\n"
        prompt += "\nAnswer:"
        return prompt
    
    def get_correct_answer(self, question: Dict[str, Any]) -> str:
        """
        Get the correct answer for a question.
        
        Args:
            question: Question dictionary from dataset
            
        Returns:
            Correct answer letter (A, B, C, or D)
        """
        idx = question['answer']
        return ['A', 'B', 'C', 'D'][idx]
    
    def get_available_subjects(self) -> List[str]:
        """
        Get list of available subjects.
        
        Returns:
            List of subject names
        """
        if self.subjects is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return self.subjects.copy()


def main():
    """Test the data loader."""
    # Initialize data loader
    loader = MMLUDataLoader()
    
    # Load a small subset for testing (using a few subjects)
    test_subjects = ['abstract_algebra', 'anatomy', 'astronomy']
    dataset = loader.load_dataset(subjects=test_subjects, split='test')
    
    # Get sample questions
    sample_questions = loader.get_sample_questions(n=3)
    
    print("\nSample Questions:")
    for i, question in enumerate(sample_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Subject: {question['subject']}")
        print(loader.format_question(question))
        print(f"Correct Answer: {loader.get_correct_answer(question)}")


if __name__ == "__main__":
    main() 