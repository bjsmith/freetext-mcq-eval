"""
Example usage of the integrated MMLU evaluation framework.
Demonstrates both custom evaluation and LM Harness integration.
"""

import json
from integrated_evaluator import IntegratedEvaluator


def example_1_lm_harness_evaluation():
    """Example 1: Evaluate a model using LM Harness."""
    print("=== Example 1: LM Harness Evaluation ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        # Evaluate GPT-2 on MMLU using LM Harness
        results = evaluator.evaluate_with_lm_harness(
            model="gpt2",
            limit=10,  # Small limit for quick testing
            device="cpu"
        )
        
        print("LM Harness Results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"LM Harness evaluation failed: {e}")
        print("This might be due to model loading issues or missing dependencies.")


def example_2_custom_framework_evaluation():
    """Example 2: Evaluate using the custom framework."""
    print("\n=== Example 2: Custom Framework Evaluation ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        # Evaluate a mock model using the custom framework
        results = evaluator.evaluate_with_custom_framework(
            model_name="mock_high_accuracy",
            model_type="mock",
            accuracy=0.85,
            subjects=["abstract_algebra", "anatomy"],
            max_questions=5
        )
        
        print("Custom Framework Results:")
        print(f"Accuracy: {results.get('accuracy', 'N/A')}")
        print(f"F1 Score: {results.get('f1_score', 'N/A')}")
        print(f"Number of Questions: {results.get('num_questions', 'N/A')}")
        
    except Exception as e:
        print(f"Custom framework evaluation failed: {e}")


def example_3_framework_comparison():
    """Example 3: Compare both frameworks on the same model."""
    print("\n=== Example 3: Framework Comparison ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        # Compare custom framework vs LM Harness
        comparison = evaluator.compare_frameworks(
            model_name="mock_high_accuracy",
            subjects=["abstract_algebra"],
            max_questions=5
        )
        
        print("Framework comparison completed!")
        print("Check the output directory for detailed results.")
        
    except Exception as e:
        print(f"Framework comparison failed: {e}")


def example_4_multi_model_evaluation():
    """Example 4: Evaluate multiple models."""
    print("\n=== Example 4: Multi-Model Evaluation ===")
    
    evaluator = IntegratedEvaluator()
    
    # Define multiple models to evaluate
    models = [
        {"model_name": "mock_high_accuracy", "accuracy": 0.9},
        {"model_name": "mock_medium_accuracy", "accuracy": 0.7},
        {"model_name": "mock_low_accuracy", "accuracy": 0.5}
    ]
    
    try:
        # Evaluate all models using the custom framework
        comparison = evaluator.evaluate_multiple_models(
            models=models,
            framework="custom",
            subjects=["abstract_algebra"],
            max_questions=5
        )
        
        print("Multi-model evaluation completed!")
        print("\nResults Summary:")
        print(comparison[['model', 'accuracy', 'f1_score']].to_string(index=False))
        
    except Exception as e:
        print(f"Multi-model evaluation failed: {e}")


def example_5_available_tasks():
    """Example 5: Get available tasks from LM Harness."""
    print("\n=== Example 5: Available Tasks ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        tasks = evaluator.get_available_tasks()
        print(f"Available tasks in LM Harness: {len(tasks)}")
        print("First 10 tasks:")
        for i, task in enumerate(tasks[:10]):
            print(f"  {i+1}. {task}")
        
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more tasks")
            
    except Exception as e:
        print(f"Failed to get available tasks: {e}")


def example_6_framework_info():
    """Example 6: Print framework comparison information."""
    print("\n=== Example 6: Framework Information ===")
    
    evaluator = IntegratedEvaluator()
    evaluator.print_framework_comparison()


def main():
    """Run all examples."""
    print("MMLU Evaluation Framework Examples")
    print("=" * 50)
    
    # Run examples
    example_1_lm_harness_evaluation()
    example_2_custom_framework_evaluation()
    example_3_framework_comparison()
    example_4_multi_model_evaluation()
    example_5_available_tasks()
    example_6_framework_info()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Check the 'integrated_results' directory for output files")
    print("2. Try evaluating your own models")
    print("3. Use the command-line interface: python integrated_evaluator.py --help")


if __name__ == "__main__":
    main() 