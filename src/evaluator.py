"""
Main evaluation engine for the framework.
Orchestrates the evaluation process using models, benchmarks, and graders.
"""

import time
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from models.base import BaseModel
from benchmarks.base import BaseBenchmark
from graders.base import BaseGrader

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluation engine."""
    
    def __init__(self, model: BaseModel, benchmark: BaseBenchmark, grader: BaseGrader):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            benchmark: Benchmark to use for evaluation
            grader: Grading algorithm to use
        """
        self.model = model
        self.benchmark = benchmark
        self.grader = grader
        
        # Results storage
        self.results = []
        self.summary = {}
        
        logger.info(f"Initialized evaluator with {model}, {benchmark}, {grader}")
    
    def evaluate(self, max_questions: Optional[int] = None, batch_size: int = 1, method: str = "logprob") -> Dict[str, Any]:
        """
        Run the evaluation.
        
        Args:
            max_questions: Maximum number of questions to evaluate (None for all)
            batch_size: Number of questions to process in each batch
            method: Evaluation method - "logprob" for log probability scoring, "direct" for direct answer prediction
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation with {self.model.model_name} on {self.benchmark.benchmark_name} using {method} method")
        
        # Load benchmark data
        self.benchmark.load_data()
        
        # Get questions
        questions = list(self.benchmark.get_questions())
        if max_questions:
            questions = questions[:max_questions]
        
        logger.info(f"Evaluating {len(questions)} questions")
        
        # Process questions
        start_time = time.time()
        
        if method == "logprob":
            for i in tqdm(range(0, len(questions), batch_size), desc="Evaluating questions"):
                batch_questions = questions[i:i + batch_size]
                self._process_batch_logprob(batch_questions)
        elif method == "direct":
            for i in tqdm(range(0, len(questions), batch_size), desc="Evaluating questions"):
                batch_questions = questions[i:i + batch_size]
                self._process_batch_direct(batch_questions)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")
        
        evaluation_time = time.time() - start_time
        
        # Generate summary
        self._generate_summary(evaluation_time)
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Accuracy: {self.summary['accuracy']:.4f}")
        
        return self.summary
    
    def _process_batch_logprob(self, questions: List[Dict[str, Any]]) -> None:
        """
        Process a batch of questions using log probability scoring.
        
        Args:
            questions: List of question data
        """
        batch_scores = []
        correct_answers = []
        
        for question_data in questions:
            try:
                # Format question
                formatted_question = self.benchmark.format_question(question_data)
                choices = self.benchmark.get_choices(question_data)
                correct_answer = self.benchmark.get_correct_answer(question_data)
                
                # Get model scores
                scores = self.model.score(formatted_question, choices)
                
                batch_scores.append(scores)
                correct_answers.append(correct_answer)
                
            except Exception as e:
                logger.warning(f"Error processing question: {e}")
                # Add dummy scores for failed questions
                choices = self.benchmark.get_choices(question_data)
                batch_scores.append([float('-inf')] * len(choices))
                correct_answers.append(self.benchmark.get_correct_answer(question_data))
        
        # Grade the batch
        batch_results = self.grader.grade_batch(batch_scores, correct_answers)
        
        # Store results with question metadata
        for i, (question_data, result) in enumerate(zip(questions, batch_results)):
            result['question'] = question_data['question']
            result['choices'] = self.benchmark.get_choices(question_data)
            result['correct_answer'] = correct_answers[i]
            result['subject'] = question_data.get('subject', 'unknown')
            result['metadata'].update(question_data.get('metadata', {}))
            result['method'] = 'logprob'
            
            self.results.append(result)
    
    def _process_batch_direct(self, questions: List[Dict[str, Any]]) -> None:
        """
        Process a batch of questions using direct answer prediction.
        
        Args:
            questions: List of question data
        """
        for question_data in questions:
            try:
                # Format question
                formatted_question = self.benchmark.format_question(question_data)
                choices = self.benchmark.get_choices(question_data)
                correct_answer = self.benchmark.get_correct_answer(question_data)
                
                # Get model prediction
                predicted_letter = self.model.predict_answer(formatted_question, choices)
                predicted_answer = ord(predicted_letter) - ord('A')  # Convert A,B,C,D to 0,1,2,3
                
                # Check if correct
                correct = predicted_answer == correct_answer
                
                result = {
                    'correct': correct,
                    'predicted_answer': predicted_answer,
                    'predicted_letter': predicted_letter,
                    'confidence': 1.0,  # Direct prediction doesn't provide confidence
                    'scores': None,
                    'probabilities': None,
                    'predicted_probability': 1.0 if correct else 0.0,
                    'correct_probability': 1.0 if correct else 0.0,
                    'question': question_data['question'],
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'subject': question_data.get('subject', 'unknown'),
                    'metadata': {
                        'grader': self.grader.grader_name,
                        'method': 'direct',
                        **question_data.get('metadata', {})
                    },
                    'method': 'direct'
                }
                
                self.results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing question: {e}")
                # Add error result
                result = {
                    'correct': False,
                    'predicted_answer': -1,
                    'predicted_letter': 'A',
                    'confidence': 0.0,
                    'scores': None,
                    'probabilities': None,
                    'predicted_probability': 0.0,
                    'correct_probability': 0.0,
                    'question': question_data['question'],
                    'choices': self.benchmark.get_choices(question_data),
                    'correct_answer': self.benchmark.get_correct_answer(question_data),
                    'subject': question_data.get('subject', 'unknown'),
                    'metadata': {
                        'grader': self.grader.grader_name,
                        'method': 'direct',
                        'error': str(e),
                        **question_data.get('metadata', {})
                    },
                    'method': 'direct'
                }
                self.results.append(result)
    
    def _process_batch(self, questions: List[Dict[str, Any]]) -> None:
        """
        Process a batch of questions (legacy method for backward compatibility).
        
        Args:
            questions: List of question data
        """
        self._process_batch_logprob(questions)
    
    def _generate_summary(self, evaluation_time: float) -> None:
        """
        Generate evaluation summary.
        
        Args:
            evaluation_time: Total evaluation time in seconds
        """
        if not self.results:
            self.summary = {
                'accuracy': 0.0,
                'total_questions': 0,
                'correct_answers': 0,
                'evaluation_time': evaluation_time,
                'questions_per_second': 0.0
            }
            return
        
        # Calculate basic metrics
        total_questions = len(self.results)
        correct_answers = sum(1 for result in self.results if result['correct'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # Calculate subject-wise accuracy
        subject_stats = {}
        for result in self.results:
            subject = result['subject']
            if subject not in subject_stats:
                subject_stats[subject] = {'total': 0, 'correct': 0}
            
            subject_stats[subject]['total'] += 1
            if result['correct']:
                subject_stats[subject]['correct'] += 1
        
        subject_accuracy = {}
        for subject, stats in subject_stats.items():
            subject_accuracy[subject] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        # Calculate confidence statistics
        confidences = [result['confidence'] for result in self.results if result['confidence'] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Calculate probability statistics
        predicted_probs = [result['predicted_probability'] for result in self.results if result['predicted_probability'] is not None]
        correct_probs = [result['correct_probability'] for result in self.results if result['correct_probability'] is not None]
        
        avg_predicted_prob = sum(predicted_probs) / len(predicted_probs) if predicted_probs else 0.0
        avg_correct_prob = sum(correct_probs) / len(correct_probs) if correct_probs else 0.0
        
        self.summary = {
            'accuracy': accuracy,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'evaluation_time': evaluation_time,
            'questions_per_second': total_questions / evaluation_time if evaluation_time > 0 else 0.0,
            'subject_accuracy': subject_accuracy,
            'subject_stats': subject_stats,
            'avg_confidence': avg_confidence,
            'avg_predicted_probability': avg_predicted_prob,
            'avg_correct_probability': avg_correct_prob,
            'model': self.model.model_name,
            'benchmark': self.benchmark.benchmark_name,
            'grader': self.grader.grader_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.model_name.replace('/', '_')
        
        # Convert results to JSON-serializable format
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, bool):
                return int(obj)  # Convert bool to int for JSON
            elif isinstance(obj, (int, float, str)) or obj is None:
                return obj
            else:
                return str(obj)  # Convert other types to string
        
        serializable_results = make_json_serializable(self.results)
        serializable_summary = make_json_serializable(self.summary)
        
        # Save detailed results
        results_file = output_dir / f"results_{model_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary
        summary_file = output_dir / f"summary_{model_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = output_dir / f"results_{model_name}_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"  - Detailed results: {results_file}")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - CSV: {csv_file}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get detailed results."""
        return self.results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        return self.summary.copy() 