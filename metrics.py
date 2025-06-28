"""
Evaluation metrics for MCQ tasks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MCQEvaluator:
    """Evaluator for Multiple Choice Question tasks."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def extract_answer(self, model_response: str) -> str:
        """
        Extract the answer from model response.
        
        Args:
            model_response: Raw model response string
            
        Returns:
            Extracted answer (A, B, C, or D)
        """
        # Clean the response
        response = model_response.strip().upper()
        
        # Look for single letter answers
        for letter in ['A', 'B', 'C', 'D']:
            # Check for isolated letter (with word boundaries)
            if f' {letter} ' in f' {response} ' or response.endswith(letter) or response.startswith(letter):
                return letter
                
        # Look for "Answer: X" pattern
        if 'ANSWER:' in response:
            answer_part = response.split('ANSWER:')[-1].strip()
            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer_part:
                    return letter
                    
        # Look for "The answer is X" pattern
        if 'THE ANSWER IS' in response:
            answer_part = response.split('THE ANSWER IS')[-1].strip()
            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer_part:
                    return letter
                    
        # If no clear answer found, return the first letter found
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter
                
        # Default to A if no answer found
        return 'A'
    
    def calculate_basic_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of correct answers
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numeric for sklearn
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        y_pred = [label_map.get(pred, 0) for pred in predictions]
        y_true = [label_map.get(gt, 0) for gt in ground_truth]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2, 3]
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
    
    def calculate_confidence_metrics(self, predictions: List[str], ground_truth: List[str], 
                                   confidences: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate confidence-based metrics.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of correct answers
            confidences: Optional list of confidence scores
            
        Returns:
            Dictionary of confidence metrics
        """
        metrics = {}
        
        if confidences is not None:
            # Calculate confidence calibration metrics
            correct = [pred == gt for pred, gt in zip(predictions, ground_truth)]
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(confidences, correct)
            metrics['expected_calibration_error'] = ece
            
            # Reliability diagram
            reliability = self._calculate_reliability(confidences, correct)
            metrics['reliability'] = reliability
            
            # Confidence vs accuracy correlation
            confidence_accuracy_corr = np.corrcoef(confidences, correct)[0, 1]
            metrics['confidence_accuracy_correlation'] = confidence_accuracy_corr
            
        return metrics
    
    def _calculate_ece(self, confidences: List[float], correct: List[bool], 
                      n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error.
        
        Args:
            confidences: List of confidence scores
            correct: List of correctness indicators
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(np.array(correct)[in_bin])
                bin_confidence = np.mean(np.array(confidences)[in_bin])
                ece += bin_size * np.abs(bin_accuracy - bin_confidence)
                
        return ece / len(confidences)
    
    def _calculate_reliability(self, confidences: List[float], correct: List[bool], 
                             n_bins: int = 10) -> Dict[str, List[float]]:
        """
        Calculate reliability diagram data.
        
        Args:
            confidences: List of confidence scores
            correct: List of correctness indicators
            n_bins: Number of bins
            
        Returns:
            Dictionary with confidence and accuracy arrays
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences_binned = []
        accuracies_binned = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(np.array(correct)[in_bin])
                bin_confidence = np.mean(np.array(confidences)[in_bin])
                
                confidences_binned.append(bin_confidence)
                accuracies_binned.append(bin_accuracy)
                
        return {
            'confidences': confidences_binned,
            'accuracies': accuracies_binned
        }
    
    def evaluate_predictions(self, predictions: List[str], ground_truth: List[str], 
                           confidences: Optional[List[float]] = None,
                           subject_breakdown: bool = True,
                           subjects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate predictions comprehensively.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of correct answers
            confidences: Optional list of confidence scores
            subject_breakdown: Whether to calculate per-subject metrics
            subjects: List of subject names for breakdown
            
        Returns:
            Comprehensive evaluation results
        """
        results = {}
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(predictions, ground_truth)
        results.update(basic_metrics)
        
        # Confidence metrics
        if confidences is not None:
            confidence_metrics = self.calculate_confidence_metrics(predictions, ground_truth, confidences)
            results.update(confidence_metrics)
        
        # Subject breakdown
        if subject_breakdown and subjects is not None:
            subject_metrics = self._calculate_subject_breakdown(predictions, ground_truth, subjects)
            results['subject_breakdown'] = subject_metrics
        
        # Confusion matrix
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        y_pred = [label_map.get(pred, 0) for pred in predictions]
        y_true = [label_map.get(gt, 0) for gt in ground_truth]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        results['confusion_matrix'] = cm
        
        self.results = results
        return results
    
    def _calculate_subject_breakdown(self, predictions: List[str], ground_truth: List[str], 
                                   subjects: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics breakdown by subject.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of correct answers
            subjects: List of subject names
            
        Returns:
            Dictionary of metrics per subject
        """
        subject_metrics = {}
        unique_subjects = list(set(subjects))
        
        for subject in unique_subjects:
            # Filter data for this subject
            subject_indices = [i for i, s in enumerate(subjects) if s == subject]
            subject_preds = [predictions[i] for i in subject_indices]
            subject_gt = [ground_truth[i] for i in subject_indices]
            
            # Calculate metrics for this subject
            metrics = self.calculate_basic_metrics(subject_preds, subject_gt)
            subject_metrics[subject] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'num_questions': len(subject_preds)
            }
            
        return subject_metrics
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot evaluation results.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_predictions() first.")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['A', 'B', 'C', 'D'], 
                   yticklabels=['A', 'B', 'C', 'D'], ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Per-class metrics
        if 'precision_per_class' in self.results:
            classes = ['A', 'B', 'C', 'D']
            precision = self.results['precision_per_class']
            recall = self.results['recall_per_class']
            f1 = self.results['f1_per_class']
            
            x = np.arange(len(classes))
            width = 0.25
            
            axes[0, 1].bar(x - width, precision, width, label='Precision')
            axes[0, 1].bar(x, recall, width, label='Recall')
            axes[0, 1].bar(x + width, f1, width, label='F1-Score')
            
            axes[0, 1].set_xlabel('Answer Class')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Per-Class Metrics')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(classes)
            axes[0, 1].legend()
        
        # 3. Subject breakdown (if available)
        if 'subject_breakdown' in self.results:
            subjects = list(self.results['subject_breakdown'].keys())
            accuracies = [self.results['subject_breakdown'][s]['accuracy'] for s in subjects]
            
            axes[1, 0].barh(subjects, accuracies)
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_title('Accuracy by Subject')
            axes[1, 0].set_xlim(0, 1)
        
        # 4. Reliability diagram (if available)
        if 'reliability' in self.results:
            reliability = self.results['reliability']
            axes[1, 1].plot(reliability['confidences'], reliability['accuracies'], 'o-', label='Model')
            axes[1, 1].plot([0, 1], [0, 1], '--', label='Perfect Calibration')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Reliability Diagram')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.results:
            raise ValueError("No results to summarize. Run evaluate_predictions() first.")
            
        print("=== MMLU Evaluation Summary ===")
        print(f"Overall Accuracy: {self.results['accuracy']:.4f}")
        print(f"Precision: {self.results['precision']:.4f}")
        print(f"Recall: {self.results['recall']:.4f}")
        print(f"F1-Score: {self.results['f1_score']:.4f}")
        
        if 'expected_calibration_error' in self.results:
            print(f"Expected Calibration Error: {self.results['expected_calibration_error']:.4f}")
            
        if 'subject_breakdown' in self.results:
            print("\n=== Subject Breakdown ===")
            for subject, metrics in self.results['subject_breakdown'].items():
                print(f"{subject}: {metrics['accuracy']:.4f} ({metrics['num_questions']} questions)")


def main():
    """Test the evaluator with dummy data."""
    evaluator = MCQEvaluator()
    
    # Dummy data
    predictions = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
    ground_truth = ['A', 'B', 'C', 'D', 'B', 'A', 'C', 'D']
    subjects = ['math', 'math', 'science', 'science', 'math', 'math', 'science', 'science']
    
    # Evaluate
    results = evaluator.evaluate_predictions(predictions, ground_truth, subjects=subjects)
    
    # Print summary
    evaluator.print_summary()


if __name__ == "__main__":
    main() 