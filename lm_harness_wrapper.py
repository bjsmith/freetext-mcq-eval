"""
Python wrapper for EleutherAI LM Evaluation Harness.
Integrates LM Harness with the existing MMLU evaluation framework.
"""

import os
import json
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LMHarnessWrapper:
    """Wrapper for EleutherAI LM Evaluation Harness."""
    
    def __init__(self, output_dir: str = "lm_harness_results"):
        """
        Initialize the LM Harness wrapper.
        
        Args:
            output_dir: Directory to store LM Harness results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def evaluate_model(self, 
                      model: str,
                      tasks: Union[str, List[str]] = "mmlu",
                      limit: Optional[int] = None,
                      device: str = "auto",
                      batch_size: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Evaluate a model using LM Harness.
        
        Args:
            model: Model name or path (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
            tasks: Task(s) to evaluate on (e.g., "mmlu", ["mmlu", "arc"])
            limit: Limit number of examples per task
            device: Device to use ("auto", "cpu", "cuda")
            batch_size: Batch size for evaluation
            **kwargs: Additional arguments for lm-eval
            
        Returns:
            Dictionary containing evaluation results
        """
        # Prepare command
        cmd = self._build_command(
            model=model,
            tasks=tasks,
            limit=limit,
            device=device,
            batch_size=batch_size,
            **kwargs
        )
        
        logger.info(f"Running LM Harness evaluation: {' '.join(cmd)}")
        
        # Run evaluation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse results
            results = self._parse_results(result.stdout)
            
            # Save results
            self._save_results(results, model, tasks)
            
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"LM Harness evaluation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def _build_command(self, 
                      model: str,
                      tasks: Union[str, List[str]],
                      limit: Optional[int] = None,
                      device: str = "auto",
                      batch_size: Optional[int] = None,
                      **kwargs) -> List[str]:
        """Build the lm-eval command."""
        cmd = ["lm-eval"]
        
        # Add model
        cmd.extend(["--model", "hf", "--model_args", f"pretrained={model}"])
        
        # Add tasks
        if isinstance(tasks, list):
            tasks_str = ",".join(tasks)
        else:
            tasks_str = tasks
        cmd.extend(["--tasks", tasks_str])
        
        # Add optional parameters
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        if device != "auto":
            cmd.extend(["--device", device])
        
        if batch_size:
            cmd.extend(["--batch_size", str(batch_size)])
        
        # Add additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def _parse_results(self, output: str) -> Dict[str, Any]:
        """Parse LM Harness output to extract results."""
        results = {}
        
        # Look for JSON results in the output
        lines = output.split('\n')
        for line in lines:
            if line.strip().startswith('{') and line.strip().endswith('}'):
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and 'results' in data:
                        results = data
                        break
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try to extract key metrics from text
        if not results:
            results = self._extract_metrics_from_text(output)
        
        return results
    
    def _extract_metrics_from_text(self, output: str) -> Dict[str, Any]:
        """Extract metrics from text output when JSON parsing fails."""
        results = {"raw_output": output}
        
        # Look for common metric patterns
        lines = output.split('\n')
        for line in lines:
            if 'accuracy' in line.lower():
                # Extract accuracy value
                import re
                match = re.search(r'accuracy[:\s]*([0-9.]+)', line.lower())
                if match:
                    results['accuracy'] = float(match.group(1))
            
            elif 'score' in line.lower():
                # Extract score value
                import re
                match = re.search(r'score[:\s]*([0-9.]+)', line.lower())
                if match:
                    results['score'] = float(match.group(1))
        
        return results
    
    def _save_results(self, results: Dict[str, Any], model: str, tasks: Union[str, List[str]]):
        """Save evaluation results to file."""
        # Create filename
        model_name = model.replace('/', '_').replace('\\', '_')
        tasks_str = "_".join(tasks) if isinstance(tasks, list) else tasks
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{tasks_str}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def evaluate_mmlu(self, 
                     model: str,
                     subjects: Optional[List[str]] = None,
                     limit: Optional[int] = None,
                     device: str = "auto",
                     **kwargs) -> Dict[str, Any]:
        """
        Evaluate a model specifically on MMLU.
        
        Args:
            model: Model name or path
            subjects: Specific MMLU subjects to evaluate on
            limit: Limit number of examples per subject
            device: Device to use
            **kwargs: Additional arguments
            
        Returns:
            MMLU evaluation results
        """
        # Build MMLU task specification
        if subjects:
            # For specific subjects, we need to use the subject names as tasks
            tasks = subjects
        else:
            # Use the general MMLU task
            tasks = "mmlu"
        
        return self.evaluate_model(
            model=model,
            tasks=tasks,
            limit=limit,
            device=device,
            **kwargs
        )
    
    def compare_models(self, 
                      models: List[str],
                      tasks: Union[str, List[str]] = "mmlu",
                      limit: Optional[int] = None,
                      **kwargs) -> pd.DataFrame:
        """
        Compare multiple models on the same tasks.
        
        Args:
            models: List of model names/paths
            tasks: Task(s) to evaluate on
            limit: Limit number of examples per task
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with comparison results
        """
        all_results = []
        
        for model in models:
            logger.info(f"Evaluating model: {model}")
            
            try:
                results = self.evaluate_model(
                    model=model,
                    tasks=tasks,
                    limit=limit,
                    **kwargs
                )
                
                # Extract key metrics
                model_results = {
                    'model': model,
                    'tasks': tasks if isinstance(tasks, str) else ','.join(tasks)
                }
                
                # Add metrics from results
                if 'results' in results:
                    for task_name, task_results in results['results'].items():
                        if isinstance(task_results, dict):
                            for metric, value in task_results.items():
                                if isinstance(value, (int, float)):
                                    model_results[f"{task_name}_{metric}"] = value
                
                all_results.append(model_results)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model}: {e}")
                all_results.append({
                    'model': model,
                    'error': str(e)
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Save comparison
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.output_dir / f"model_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        logger.info(f"Model comparison saved to: {comparison_file}")
        
        return comparison_df
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks in LM Harness."""
        try:
            result = subprocess.run(
                ["lm-eval", "--tasks", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse task list from output
            tasks = []
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('Available'):
                    tasks.append(line.strip())
            
            return tasks
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get available tasks: {e}")
            return []
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about a specific task."""
        try:
            result = subprocess.run(
                ["lm-eval", "--tasks", task_name, "--task_args", "info"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Try to parse as JSON
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw_output": result.stdout}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get task info for {task_name}: {e}")
            return {"error": str(e)}


def main():
    """Example usage of the LM Harness wrapper."""
    wrapper = LMHarnessWrapper()
    
    # Example 1: Evaluate a single model on MMLU
    print("=== Example 1: Single Model MMLU Evaluation ===")
    try:
        results = wrapper.evaluate_mmlu(
            model="gpt2",
            limit=10,  # Limit for quick testing
            device="cpu"
        )
        print(f"Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    # Example 2: Compare multiple models
    print("\n=== Example 2: Model Comparison ===")
    models = ["gpt2", "gpt2-medium"]
    try:
        comparison = wrapper.compare_models(
            models=models,
            tasks="mmlu",
            limit=5  # Small limit for quick testing
        )
        print("Model Comparison:")
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"Comparison failed: {e}")
    
    # Example 3: Get available tasks
    print("\n=== Example 3: Available Tasks ===")
    tasks = wrapper.get_available_tasks()
    print(f"Available tasks: {tasks[:10]}...")  # Show first 10


if __name__ == "__main__":
    main() 