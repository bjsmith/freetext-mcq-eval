"""
Command-line interface for the evaluation framework.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import click
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.openai_model import OpenAIModel
from benchmarks.mmlu import MMLUBenchmark
from graders.logprob_grader import LogProbGrader
from evaluator import Evaluator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
def cli():
    """FreeText MCQ Evaluation Framework CLI."""
    pass


@cli.command()
@click.option('--model', default='gpt-4o-mini', help='Model name to evaluate')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--subjects', multiple=True, help='MMLU subjects to evaluate (can specify multiple)')
@click.option('--max-questions', type=int, help='Maximum number of questions to evaluate')
@click.option('--output-dir', default='./results', help='Output directory for results')
@click.option('--batch-size', default=1, help='Batch size for processing')
def evaluate(model: str, api_key: str, subjects: tuple, max_questions: Optional[int], 
            output_dir: str, batch_size: int):
    """Evaluate a model on MMLU benchmark."""
    
    if not api_key:
        click.echo("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key.")
        return
    
    try:
        # Initialize components
        click.echo(f"Initializing model: {model}")
        model_instance = OpenAIModel(model_name=model, api_key=api_key)
        
        click.echo("Initializing MMLU benchmark")
        subjects_list = list(subjects) if subjects else None
        benchmark = MMLUBenchmark(subjects=subjects_list)
        
        click.echo("Initializing log probability grader")
        grader = LogProbGrader()
        
        # Create evaluator
        evaluator = Evaluator(model_instance, benchmark, grader)
        
        # Run evaluation
        click.echo("Starting evaluation...")
        summary = evaluator.evaluate(max_questions=max_questions, batch_size=batch_size)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("EVALUATION RESULTS")
        click.echo("="*50)
        click.echo(f"Model: {summary['model']}")
        click.echo(f"Benchmark: {summary['benchmark']}")
        click.echo(f"Grader: {summary['grader']}")
        click.echo(f"Total Questions: {summary['total_questions']}")
        click.echo(f"Correct Answers: {summary['correct_answers']}")
        click.echo(f"Accuracy: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
        click.echo(f"Evaluation Time: {summary['evaluation_time']:.2f} seconds")
        click.echo(f"Questions/Second: {summary['questions_per_second']:.2f}")
        click.echo(f"Average Confidence: {summary['avg_confidence']:.4f}")
        
        # Subject-wise results
        if summary['subject_accuracy']:
            click.echo("\nSubject-wise Accuracy:")
            for subject, acc in sorted(summary['subject_accuracy'].items()):
                click.echo(f"  {subject}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Save results
        evaluator.save_results(output_dir)
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")
        logging.exception("Evaluation failed")


@cli.command()
def list_subjects():
    """List available MMLU subjects."""
    try:
        benchmark = MMLUBenchmark()
        subjects = benchmark.get_subjects()
        
        click.echo("Available MMLU subjects:")
        for subject in subjects:
            click.echo(f"  - {subject}")
        
    except Exception as e:
        click.echo(f"Error listing subjects: {e}")


@cli.command()
@click.option('--subjects', multiple=True, help='Subjects to get stats for')
def subject_stats(subjects: tuple):
    """Get statistics for MMLU subjects."""
    try:
        subjects_list = list(subjects) if subjects else None
        benchmark = MMLUBenchmark(subjects=subjects_list)
        benchmark.load_data()
        
        stats = benchmark.get_subject_stats()
        
        click.echo("MMLU Subject Statistics:")
        total_questions = sum(stats.values())
        click.echo(f"Total Questions: {total_questions}")
        click.echo("\nQuestions per Subject:")
        
        for subject, count in sorted(stats.items()):
            percentage = (count / total_questions) * 100 if total_questions > 0 else 0
            click.echo(f"  {subject}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        click.echo(f"Error getting subject stats: {e}")


if __name__ == '__main__':
    cli() 