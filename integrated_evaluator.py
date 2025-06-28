"""
Integrated evaluator that combines custom MMLU evaluation with LM Harness.
Provides a unified interface for both evaluation approaches.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import custom components
from data_loader import MMLUDataLoader
from model_interface import create_model_interface
from metrics import MCQEvaluator
from evaluate_mmlu import MMLUEvaluator

# Import LM Harness wrapper
from lm_harness_wrapper import LMHarnessWrapper


class IntegratedEvaluator:
    """Integrated evaluator combining custom framework with LM Harness."""
    
    def __init__(self, output_dir: str = "integrated_results"):
        """
        Initialize the integrated evaluator.
        
        Args:
            output_dir: Directory to store all results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.custom_evaluator = MMLUEvaluator(output_dir=str(self.output_dir / "custom"))
        self.lm_harness = LMHarnessWrapper(output_dir=str(self.output_dir / "lm_harness"))
        
    def evaluate_with_custom_framework(self, 
                                     model_name: str,
                                     model_type: str = "mock",
                                     subjects: Optional[List[str]] = None,
                                     max_questions: Optional[int] = None,
                                     **kwargs) -> Dict[str, Any]:
        """
        Evaluate using the custom framework.
        
        Args:
            model_name: Name of the model
            model_type: Type of model interface
            subjects: List of subjects to evaluate on
            max_questions: Maximum questions per subject
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results
        """
        print("=== Using Custom Evaluation Framework ===")
        
        results = self.custom_evaluator.evaluate_model(
            model_name=model_name,
            model_type=model_type,
            subjects=subjects,
            max_questions=max_questions,
            **kwargs
        )
        
        # Add framework identifier
        results['framework'] = 'custom'
        results['evaluation_timestamp'] = datetime.now().isoformat()
        
        return results
    
    def evaluate_with_lm_harness(self,
                               model: str,
                               subjects: Optional[List[str]] = None,
                               limit: Optional[int] = None,
                               device: str = "auto",
                               **kwargs) -> Dict[str, Any]:
        """
        Evaluate using LM Harness.
        
        Args:
            model: Model name or path
            subjects: Specific MMLU subjects to evaluate on
            limit: Limit number of examples per task
            device: Device to use
            **kwargs: Additional arguments
            
        Returns:
            LM Harness evaluation results
        """
        print("=== Using LM Harness Framework ===")
        
        results = self.lm_harness.evaluate_mmlu(
            model=model,
            subjects=subjects,
            limit=limit,
            device=device,
            **kwargs
        )
        
        # Add framework identifier
        results['framework'] = 'lm_harness'
        results['evaluation_timestamp'] = datetime.now().isoformat()
        
        return results
    
    def compare_frameworks(self,
                          model_name: str,
                          subjects: Optional[List[str]] = None,
                          max_questions: Optional[int] = None,
                          device: str = "auto",
                          **kwargs) -> Dict[str, Any]:
        """
        Compare results from both frameworks on the same model.
        
        Args:
            model_name: Name of the model
            subjects: List of subjects to evaluate on
            max_questions: Maximum questions per subject (custom framework)
            device: Device to use (LM Harness)
            **kwargs: Additional arguments
            
        Returns:
            Comparison results
        """
        print("=== Comparing Custom vs LM Harness Frameworks ===")
        
        comparison = {
            'model': model_name,
            'subjects': subjects,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # Evaluate with custom framework
        try:
            print("Running custom framework evaluation...")
            custom_results = self.evaluate_with_custom_framework(
                model_name=model_name,
                subjects=subjects,
                max_questions=max_questions,
                **kwargs
            )
            comparison['custom_results'] = custom_results
        except Exception as e:
            print(f"Custom framework evaluation failed: {e}")
            comparison['custom_results'] = {'error': str(e)}
        
        # Evaluate with LM Harness
        try:
            print("Running LM Harness evaluation...")
            lm_results = self.evaluate_with_lm_harness(
                model=model_name,
                subjects=subjects,
                limit=max_questions,
                device=device,
                **kwargs
            )
            comparison['lm_harness_results'] = lm_results
        except Exception as e:
            print(f"LM Harness evaluation failed: {e}")
            comparison['lm_harness_results'] = {'error': str(e)}
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.output_dir / f"framework_comparison_{model_name.replace('/', '_')}_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        comparison_serializable = convert_numpy(comparison)
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_serializable, f, indent=2)
        
        print(f"Framework comparison saved to: {comparison_file}")
        
        return comparison
    
    def evaluate_multiple_models(self,
                               models: List[Dict[str, Any]],
                               framework: str = "lm_harness",
                               subjects: Optional[List[str]] = None,
                               **kwargs) -> pd.DataFrame:
        """
        Evaluate multiple models using the specified framework.
        
        Args:
            models: List of model configurations
            framework: Framework to use ("custom", "lm_harness", or "both")
            subjects: List of subjects to evaluate on
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with evaluation results
        """
        all_results = []
        
        for model_config in models:
            model_name = model_config.get('model_name', model_config.get('model'))
            print(f"\nEvaluating model: {model_name}")
            
            try:
                if framework == "custom":
                    # Remove model_name from config to avoid duplicate argument
                    config_copy = model_config.copy()
                    config_copy.pop('model_name', None)
                    
                    results = self.evaluate_with_custom_framework(
                        model_name=model_name,
                        **config_copy,
                        subjects=subjects,
                        **kwargs
                    )
                    
                elif framework == "lm_harness":
                    # Remove model from config to avoid duplicate argument
                    config_copy = model_config.copy()
                    config_copy.pop('model', None)
                    
                    results = self.evaluate_with_lm_harness(
                        model=model_name,
                        subjects=subjects,
                        **config_copy,
                        **kwargs
                    )
                    
                elif framework == "both":
                    # Remove model_name from config to avoid duplicate argument
                    config_copy = model_config.copy()
                    config_copy.pop('model_name', None)
                    
                    results = self.compare_frameworks(
                        model_name=model_name,
                        subjects=subjects,
                        **config_copy,
                        **kwargs
                    )
                    
                else:
                    raise ValueError(f"Unknown framework: {framework}")
                
                # Extract key metrics for DataFrame
                summary = self._extract_summary_metrics(results, framework)
                all_results.append(summary)
                
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
                all_results.append({
                    'model': model_name,
                    'framework': framework,
                    'error': str(e)
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.output_dir / f"multi_model_comparison_{framework}_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\nMulti-model comparison saved to: {comparison_file}")
        
        return comparison_df
    
    def _extract_summary_metrics(self, results: Dict[str, Any], framework: str) -> Dict[str, Any]:
        """Extract summary metrics from evaluation results."""
        summary = {
            'framework': framework,
            'evaluation_timestamp': results.get('evaluation_timestamp', '')
        }
        
        if framework == "custom":
            # Extract from custom framework results
            summary['model'] = results.get('model_name', '')
            summary['accuracy'] = results.get('accuracy', 0.0)
            summary['precision'] = results.get('precision', 0.0)
            summary['recall'] = results.get('recall', 0.0)
            summary['f1_score'] = results.get('f1_score', 0.0)
            summary['num_questions'] = results.get('num_questions', 0)
            
        elif framework == "lm_harness":
            # Extract from LM Harness results
            summary['model'] = results.get('model', '')
            if 'results' in results:
                for task_name, task_results in results['results'].items():
                    if isinstance(task_results, dict):
                        for metric, value in task_results.items():
                            if isinstance(value, (int, float)):
                                summary[f"{task_name}_{metric}"] = value
                                
        elif framework == "both":
            # Extract from framework comparison
            summary['model'] = results.get('model', '')
            
            # Custom results
            custom_results = results.get('custom_results', {})
            if 'error' not in custom_results:
                summary['custom_accuracy'] = custom_results.get('accuracy', 0.0)
                summary['custom_f1_score'] = custom_results.get('f1_score', 0.0)
            
            # LM Harness results
            lm_results = results.get('lm_harness_results', {})
            if 'error' not in lm_results and 'results' in lm_results:
                for task_name, task_results in lm_results['results'].items():
                    if isinstance(task_results, dict):
                        for metric, value in task_results.items():
                            if isinstance(value, (int, float)):
                                summary[f"lm_{task_name}_{metric}"] = value
        
        return summary
    
    def get_available_tasks(self) -> List[str]:
        """Get available tasks from LM Harness."""
        return self.lm_harness.get_available_tasks()
    
    def print_framework_comparison(self):
        """Print a comparison of the two frameworks."""
        print("=== Framework Comparison ===")
        print("\nCustom Framework:")
        print("- Pros: Full control, custom metrics, detailed analysis")
        print("- Cons: Requires model integration, more setup")
        print("- Best for: Research, custom metrics, detailed analysis")
        
        print("\nLM Harness Framework:")
        print("- Pros: Standardized, many tasks, optimized performance")
        print("- Cons: Less flexibility, limited to supported models")
        print("- Best for: Standard evaluations, benchmarking, production")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Integrated MMLU evaluation with custom framework and LM Harness")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--framework", type=str, default="lm_harness", 
                       choices=["custom", "lm_harness", "both"], 
                       help="Evaluation framework to use")
    parser.add_argument("--subjects", nargs="+", help="MMLU subjects to evaluate on")
    parser.add_argument("--max-questions", type=int, help="Maximum questions per subject")
    parser.add_argument("--output-dir", type=str, default="integrated_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--model-type", type=str, default="mock", help="Model type for custom framework")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = IntegratedEvaluator(output_dir=args.output_dir)
    
    if args.framework == "custom":
        results = evaluator.evaluate_with_custom_framework(
            model_name=args.model,
            model_type=args.model_type,
            subjects=args.subjects,
            max_questions=args.max_questions,
            device=args.device
        )
        
    elif args.framework == "lm_harness":
        results = evaluator.evaluate_with_lm_harness(
            model=args.model,
            subjects=args.subjects,
            limit=args.max_questions,
            device=args.device
        )
        
    elif args.framework == "both":
        results = evaluator.compare_frameworks(
            model_name=args.model,
            subjects=args.subjects,
            max_questions=args.max_questions,
            device=args.device
        )
    
    print(f"\nEvaluation completed for {args.model} using {args.framework} framework")


if __name__ == "__main__":
    # Example usage
    evaluator = IntegratedEvaluator()
    
    # Example 1: Evaluate with LM Harness
    print("=== Example 1: LM Harness Evaluation ===")
    try:
        results = evaluator.evaluate_with_lm_harness(
            model="gpt2",
            limit=10,  # Small limit for quick testing
            device="cpu"
        )
        print(f"LM Harness Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        print(f"LM Harness evaluation failed: {e}")
    
    # Example 2: Compare frameworks
    print("\n=== Example 2: Framework Comparison ===")
    try:
        comparison = evaluator.compare_frameworks(
            model_name="mock_high_accuracy",
            subjects=["abstract_algebra"],
            max_questions=5
        )
        print("Framework comparison completed")
    except Exception as e:
        print(f"Framework comparison failed: {e}")
    
    # Example 3: Multi-model evaluation
    print("\n=== Example 3: Multi-Model Evaluation ===")
    models = [
        {"model_name": "mock_high_accuracy", "accuracy": 0.9},
        {"model_name": "mock_medium_accuracy", "accuracy": 0.7}
    ]
    
    try:
        comparison = evaluator.evaluate_multiple_models(
            models=models,
            framework="custom",
            subjects=["abstract_algebra"],
            max_questions=5
        )
        print("Multi-model evaluation completed")
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"Multi-model evaluation failed: {e}") 