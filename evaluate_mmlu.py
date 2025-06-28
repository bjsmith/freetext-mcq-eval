"""
Main script for evaluating models on MMLU dataset.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from data_loader import MMLUDataLoader
from model_interface import create_model_interface
from metrics import MCQEvaluator


class MMLUEvaluator:
    """Main evaluator class for MMLU."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the MMLU evaluator.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.data_loader = MMLUDataLoader()
        self.evaluator = MCQEvaluator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_model(self, model_name: str, model_type: str = "mock", 
                      subjects: Optional[List[str]] = None, 
                      max_questions: Optional[int] = None,
                      **model_kwargs) -> Dict[str, Any]:
        """
        Evaluate a model on MMLU.
        
        Args:
            model_name: Name of the model
            model_type: Type of model interface
            subjects: List of subjects to evaluate on
            max_questions: Maximum number of questions per subject
            **model_kwargs: Additional arguments for model interface
            
        Returns:
            Evaluation results
        """
        print(f"=== Evaluating {model_name} on MMLU ===")
        
        # Load dataset
        dataset = self.data_loader.load_dataset(subjects=subjects, split='test')
        
        # Create model interface
        model = create_model_interface(model_name, model_type, **model_kwargs)
        model.load_model()
        
        # Prepare questions
        questions = []
        ground_truth = []
        question_subjects = []
        
        if subjects:
            # Filter by subjects
            for subject in subjects:
                subject_data = self.data_loader.get_subject_data(subject)
                
                # Limit questions per subject
                if max_questions:
                    subject_data = subject_data.head(max_questions)
                
                for _, row in subject_data.iterrows():
                    formatted_question = self.data_loader.format_question(row)
                    questions.append(formatted_question)
                    ground_truth.append(self.data_loader.get_correct_answer(row))
                    question_subjects.append(subject)
        else:
            # Use all data
            if max_questions:
                dataset = dataset.select(range(min(max_questions, len(dataset))))
            
            for item in dataset:
                formatted_question = self.data_loader.format_question(item)
                questions.append(formatted_question)
                ground_truth.append(self.data_loader.get_correct_answer(item))
                question_subjects.append(item['subject'])
        
        print(f"Evaluating on {len(questions)} questions")
        
        # Generate predictions
        predictions = model.generate_answers_batch(questions)
        
        # Extract answers from model responses
        extracted_predictions = []
        for pred in predictions:
            extracted_pred = self.evaluator.extract_answer(pred)
            extracted_predictions.append(extracted_pred)
        
        # Evaluate predictions
        results = self.evaluator.evaluate_predictions(
            predictions=extracted_predictions,
            ground_truth=ground_truth,
            subjects=question_subjects
        )
        
        # Add metadata
        results['model_name'] = model_name
        results['model_type'] = model_type
        results['num_questions'] = len(questions)
        results['subjects'] = subjects if subjects else list(set(question_subjects))
        results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        self._save_results(results, model_name)
        
        # Print summary
        self.evaluator.print_summary()
        
        return results
    
    def _save_results(self, results: Dict[str, Any], model_name: str):
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """
        Prepare object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj
    
    def compare_models(self, model_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_configs: List of model configurations
            
        Returns:
            DataFrame with comparison results
        """
        all_results = []
        
        for config in model_configs:
            print(f"\n{'='*50}")
            results = self.evaluate_model(**config)
            
            # Extract key metrics
            summary = {
                'model_name': results['model_name'],
                'model_type': results['model_type'],
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'num_questions': results['num_questions']
            }
            
            all_results.append(summary)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(self.output_dir, f"model_comparison_{timestamp}.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\nModel comparison saved to: {comparison_file}")
        print("\nModel Comparison Summary:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate models on MMLU dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model-type", type=str, default="mock", 
                       choices=["mock", "huggingface", "pipeline"], 
                       help="Type of model interface")
    parser.add_argument("--subjects", nargs="+", help="Subjects to evaluate on")
    parser.add_argument("--max-questions", type=int, help="Maximum questions per subject")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MMLUEvaluator(output_dir=args.output_dir)
    
    # Evaluate model
    results = evaluator.evaluate_model(
        model_name=args.model,
        model_type=args.model_type,
        subjects=args.subjects,
        max_questions=args.max_questions,
        device=args.device
    )
    
    print(f"\nEvaluation completed for {args.model}")


if __name__ == "__main__":
    # Example usage
    evaluator = MMLUEvaluator()
    
    # Test with mock models
    mock_configs = [
        {
            'model_name': 'mock_high_accuracy',
            'model_type': 'mock',
            'accuracy': 0.9,
            'subjects': ['abstract_algebra', 'anatomy'],
            'max_questions': 10
        },
        {
            'model_name': 'mock_medium_accuracy',
            'model_type': 'mock',
            'accuracy': 0.7,
            'subjects': ['abstract_algebra', 'anatomy'],
            'max_questions': 10
        },
        {
            'model_name': 'mock_low_accuracy',
            'model_type': 'mock',
            'accuracy': 0.5,
            'subjects': ['abstract_algebra', 'anatomy'],
            'max_questions': 10
        }
    ]
    
    # Compare models
    comparison = evaluator.compare_models(mock_configs)
    
    # Example of evaluating a real model (commented out)
    # real_model_config = {
    #     'model_name': 'gpt2',
    #     'model_type': 'huggingface',
    #     'subjects': ['abstract_algebra'],
    #     'max_questions': 5
    # }
    # evaluator.evaluate_model(**real_model_config) 