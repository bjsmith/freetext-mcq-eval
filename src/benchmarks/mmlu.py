"""
MMLU (Massive Multitask Language Understanding) benchmark implementation.
Loads data from HuggingFace datasets.
"""

from typing import List, Dict, Any, Optional, Iterator
from datasets import load_dataset
import logging

from benchmarks.base import BaseBenchmark

logger = logging.getLogger(__name__)


class MMLUBenchmark(BaseBenchmark):
    """MMLU benchmark implementation."""
    
    def __init__(self, benchmark_name: str = "mmlu", subjects: Optional[List[str]] = None, **kwargs):
        """
        Initialize MMLU benchmark.
        
        Args:
            benchmark_name: Benchmark name
            subjects: List of subjects to include (if None, includes all)
            **kwargs: Additional parameters
        """
        super().__init__(benchmark_name, **kwargs)
        self.subjects = subjects
        self.dataset = None
        self.questions = []
        
        # MMLU subjects available
        self.available_subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
            'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
            'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
            'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions'
        ]
    
    def load_data(self) -> None:
        """Load MMLU data from HuggingFace."""
        try:
            logger.info("Loading MMLU dataset from HuggingFace...")
            
            # Load the dataset
            self.dataset = load_dataset("cais/mmlu", "all")
            
            # Filter by subjects if specified
            if self.subjects:
                # Validate subjects
                invalid_subjects = [s for s in self.subjects if s not in self.available_subjects]
                if invalid_subjects:
                    raise ValueError(f"Invalid subjects: {invalid_subjects}")
                
                # Filter dataset by subjects
                self.dataset = self.dataset.filter(
                    lambda x: x['subject'] in self.subjects
                )
            
            logger.info(f"Loaded {len(self.dataset['test'])} questions from MMLU")
            
        except Exception as e:
            logger.error(f"Error loading MMLU dataset: {e}")
            raise
    
    def get_questions(self) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator over all questions in the benchmark.
        
        Yields:
            Dictionary containing question data
        """
        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        for item in self.dataset['test']:
            yield {
                'question': item['question'],
                'choices': item['choices'],  # Already a list
                'correct_answer': item['answer'],  # Already 0-based index
                'subject': item['subject'],
                'metadata': {
                    'id': item.get('id', ''),
                    'source': 'mmlu'
                }
            }
    
    def format_question(self, question_data: Dict[str, Any]) -> str:
        """
        Format a question for model input.
        
        Args:
            question_data: Question data from get_questions()
            
        Returns:
            Formatted question string
        """
        question = question_data['question']
        choices = question_data['choices']
        
        # Format as multiple choice question
        formatted = f"Question: {question}\n\n"
        formatted += "Choose the best answer from the following options:\n"
        
        for i, choice in enumerate(choices):
            formatted += f"{chr(65 + i)}. {choice}\n"
        
        formatted += "\nAnswer:"
        
        return formatted
    
    def get_choices(self, question_data: Dict[str, Any]) -> List[str]:
        """
        Get the choices for a question.
        
        Args:
            question_data: Question data from get_questions()
            
        Returns:
            List of choice strings
        """
        return question_data['choices']
    
    def get_correct_answer(self, question_data: Dict[str, Any]) -> int:
        """
        Get the correct answer index for a question.
        
        Args:
            question_data: Question data from get_questions()
            
        Returns:
            Index of correct answer
        """
        return question_data['correct_answer']
    
    def get_subjects(self) -> List[str]:
        """Get list of available subjects."""
        return self.available_subjects.copy()
    
    def get_subject_stats(self) -> Dict[str, int]:
        """Get statistics about questions per subject."""
        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        stats = {}
        for item in self.dataset['test']:
            subject = item['subject']
            stats[subject] = stats.get(subject, 0) + 1
        
        return stats 