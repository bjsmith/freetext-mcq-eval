# MMLU Evaluation Framework

A comprehensive framework for evaluating Language Model performance on the Massive Multitask Language Understanding (MMLU) dataset. This project provides tools to compare different metrics and evaluate various LLM models on MCQ tasks, with integration of both custom evaluation and the EleutherAI LM Evaluation Harness.

## Features

- **MMLU Dataset Integration**: Easy loading and preprocessing of the MMLU dataset from Hugging Face
- **Multiple Model Interfaces**: Support for Hugging Face models, pipeline models, and mock models for testing
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confidence calibration, and subject breakdown
- **LM Harness Integration**: Full integration with EleutherAI LM Evaluation Harness for standardized evaluations
- **Flexible Evaluation**: Evaluate on specific subjects or the entire dataset
- **Results Visualization**: Automatic plotting of confusion matrices, reliability diagrams, and subject comparisons
- **Batch Processing**: Efficient batch evaluation of multiple models
- **Results Export**: Save results in JSON and CSV formats
- **Framework Comparison**: Compare results between custom framework and LM Harness

## Project Structure

```
freetext-mcq-eval/
├── data_loader.py              # MMLU dataset loading and preprocessing
├── model_interface.py          # Model interfaces for different LLM types
├── metrics.py                 # Evaluation metrics and visualization
├── evaluate_mmlu.py           # Main evaluation script (custom framework)
├── lm_harness_wrapper.py      # Python wrapper for LM Evaluation Harness
├── integrated_evaluator.py    # Integrated evaluator combining both frameworks
├── example_usage.py           # Example usage scripts
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── venv/                     # Python virtual environment
├── results/                  # Output directory for custom evaluation results
├── integrated_results/       # Output directory for integrated evaluation results
└── lm_harness_results/       # Output directory for LM Harness results
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Run the Example Script

The easiest way to get started is to run the example script:

```bash
python example_usage.py
```

This will demonstrate:
- LM Harness evaluation on GPT-2
- Custom framework evaluation with mock models
- Framework comparison
- Multi-model evaluation
- Available tasks from LM Harness

### 2. Use the Integrated Evaluator

```python
from integrated_evaluator import IntegratedEvaluator

# Initialize the integrated evaluator
evaluator = IntegratedEvaluator()

# Evaluate using LM Harness (recommended for most cases)
results = evaluator.evaluate_with_lm_harness(
    model="gpt2",
    limit=100,  # Number of examples to evaluate
    device="cpu"
)

# Evaluate using custom framework
results = evaluator.evaluate_with_custom_framework(
    model_name="my_model",
    model_type="huggingface",
    subjects=["abstract_algebra", "anatomy"],
    max_questions=50
)

# Compare both frameworks
comparison = evaluator.compare_frameworks(
    model_name="gpt2",
    subjects=["abstract_algebra"],
    max_questions=20
)
```

### 3. Command Line Usage

```bash
# Evaluate with LM Harness (recommended)
python integrated_evaluator.py --model gpt2 --framework lm_harness --max-questions 50

# Evaluate with custom framework
python integrated_evaluator.py --model gpt2 --framework custom --model-type huggingface --max-questions 50

# Compare both frameworks
python integrated_evaluator.py --model gpt2 --framework both --max-questions 20
```

## Framework Comparison

### LM Harness Framework (Recommended)
- **Pros**: Standardized evaluation, optimized performance, many supported tasks, production-ready
- **Cons**: Less flexibility for custom metrics, limited to supported models
- **Best for**: Standard evaluations, benchmarking, production use

### Custom Framework
- **Pros**: Full control, custom metrics, detailed analysis, any model support
- **Cons**: Requires more setup, less optimized
- **Best for**: Research, custom metrics, detailed analysis

## Available Subjects

The MMLU dataset includes 57 subjects across various domains:

**Humanities:**
- abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge, college_biology, college_chemistry, college_computer_science, college_mathematics, college_medicine, college_physics, computer_security, conceptual_physics, econometrics, electrical_engineering, elementary_mathematics, formal_logic, global_facts, high_school_biology, high_school_chemistry, high_school_computer_science, high_school_european_history, high_school_geography, high_school_government_and_politics, high_school_macroeconomics, high_school_mathematics, high_school_microeconomics, high_school_physics, high_school_psychology, high_school_statistics, high_school_us_history, high_school_world_history, human_aging, human_sexuality, international_law, jurisprudence, logical_fallacies, machine_learning, management, marketing, medical_genetics, miscellaneous, moral_disputes, moral_scenarios, nutrition, philosophy, prehistory, professional_accounting, professional_law, professional_medicine, professional_psychology, public_relations, security_studies, sociology, us_foreign_policy, virology, world_religions

## Model Types

### 1. Mock Models (Custom Framework)
For testing and development:
```python
model_config = {
    'model_name': 'test_model',
    'model_type': 'mock',
    'accuracy': 0.8  # Simulated accuracy
}
```

### 2. Hugging Face Models
For local models:
```python
model_config = {
    'model_name': 'gpt2',
    'model_type': 'huggingface',
    'device': 'auto',  # or 'cpu', 'cuda'
    'max_length': 512
}
```

### 3. LM Harness Models
For LM Harness evaluation:
```python
# Direct model names work with LM Harness
models = ["gpt2", "gpt2-medium", "meta-llama/Llama-2-7b-hf"]
```

## Evaluation Metrics

The framework calculates the following metrics:

### Basic Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Per-class metrics**: Metrics for each answer choice (A, B, C, D)

### Advanced Metrics
- **Confusion Matrix**: Detailed breakdown of predictions vs. actual answers
- **Subject Breakdown**: Performance across different subjects
- **Confidence Calibration**: Expected Calibration Error (ECE)
- **Reliability Diagram**: Confidence vs. accuracy relationship

## Output Files

The framework generates several output files:

1. **Individual Results** (`model_name_timestamp.json`):
   - Complete evaluation results
   - All metrics and metadata
   - Raw predictions and ground truth

2. **Model Comparison** (`model_comparison_timestamp.csv`):
   - Summary comparison of multiple models
   - Key metrics in tabular format

3. **Framework Comparison** (`framework_comparison_timestamp.json`):
   - Comparison between custom framework and LM Harness
   - Results from both evaluation approaches

4. **Visualizations** (displayed automatically):
   - Confusion matrices
   - Per-class metric comparisons
   - Subject breakdown charts
   - Reliability diagrams

## Advanced Usage

### Multi-Model Evaluation

```python
# Evaluate multiple models
models = [
    {"model_name": "gpt2", "model_type": "huggingface"},
    {"model_name": "gpt2-medium", "model_type": "huggingface"},
    {"model_name": "mock_high_accuracy", "model_type": "mock", "accuracy": 0.9}
]

comparison = evaluator.evaluate_multiple_models(
    models=models,
    framework="custom",  # or "lm_harness" or "both"
    subjects=["abstract_algebra", "anatomy"],
    max_questions=20
)
```

### Custom Metrics Integration

```python
# Add custom metrics to the evaluation
from metrics import MCQEvaluator

evaluator = MCQEvaluator()
# Add your custom metrics here
```

### LM Harness Task Discovery

```python
# Get available tasks from LM Harness
tasks = evaluator.get_available_tasks()
print(f"Available tasks: {tasks}")

# Get information about a specific task
task_info = evaluator.lm_harness.get_task_info("mmlu")
print(f"MMLU task info: {task_info}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check model name and ensure it's available on Hugging Face
3. **Dataset Loading Issues**: Check internet connection for Hugging Face dataset download
4. **LM Harness Errors**: Ensure lm-eval is properly installed and accessible

### Performance Tips

1. **Use GPU**: Set `device='cuda'` for faster inference
2. **Batch Processing**: Use `generate_answers_batch()` for efficiency
3. **Limit Questions**: Use `max_questions` parameter for quick testing
4. **Subject Filtering**: Evaluate on specific subjects to reduce computation time
5. **Use LM Harness**: For production evaluations, prefer LM Harness for better performance

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{mmlu_eval_framework,
  title={MMLU Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/freetext-mcq-eval}
}
```

## Acknowledgments

- MMLU dataset creators
- EleutherAI for the LM Evaluation Harness
- Hugging Face for the datasets and transformers libraries
- The open-source community for various tools and libraries used in this project 