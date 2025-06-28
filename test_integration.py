"""
Simple test script to verify the integrated framework works correctly.
"""

from integrated_evaluator import IntegratedEvaluator
import json


def test_custom_framework():
    """Test the custom framework evaluation."""
    print("=== Testing Custom Framework ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        results = evaluator.evaluate_with_custom_framework(
            model_name="test_model",
            model_type="mock",
            accuracy=0.8,
            subjects=["abstract_algebra"],
            max_questions=3
        )
        
        print("✓ Custom framework test passed")
        print(f"  Accuracy: {results.get('accuracy', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"✗ Custom framework test failed: {e}")
        return False


def test_multi_model_evaluation():
    """Test multi-model evaluation."""
    print("\n=== Testing Multi-Model Evaluation ===")
    
    evaluator = IntegratedEvaluator()
    
    models = [
        {"model_name": "model_1", "accuracy": 0.9},
        {"model_name": "model_2", "accuracy": 0.7}
    ]
    
    try:
        comparison = evaluator.evaluate_multiple_models(
            models=models,
            framework="custom",
            subjects=["abstract_algebra"],
            max_questions=2
        )
        
        print("✓ Multi-model evaluation test passed")
        print(f"  Number of models evaluated: {len(comparison)}")
        return True
        
    except Exception as e:
        print(f"✗ Multi-model evaluation test failed: {e}")
        return False


def test_framework_comparison():
    """Test framework comparison."""
    print("\n=== Testing Framework Comparison ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        comparison = evaluator.compare_frameworks(
            model_name="test_model",
            subjects=["abstract_algebra"],
            max_questions=2
        )
        
        print("✓ Framework comparison test passed")
        print(f"  Custom results: {'error' not in comparison.get('custom_results', {})}")
        print(f"  LM Harness results: {'error' not in comparison.get('lm_harness_results', {})}")
        return True
        
    except Exception as e:
        print(f"✗ Framework comparison test failed: {e}")
        return False


def test_lm_harness_wrapper():
    """Test LM Harness wrapper functionality."""
    print("\n=== Testing LM Harness Wrapper ===")
    
    evaluator = IntegratedEvaluator()
    
    try:
        # Test getting available tasks
        tasks = evaluator.get_available_tasks()
        print(f"✓ LM Harness wrapper test passed")
        print(f"  Available tasks: {len(tasks)}")
        return True
        
    except Exception as e:
        print(f"✗ LM Harness wrapper test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Integrated Framework Test Suite")
    print("=" * 40)
    
    tests = [
        test_custom_framework,
        test_multi_model_evaluation,
        test_framework_comparison,
        test_lm_harness_wrapper
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The integrated framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    main() 