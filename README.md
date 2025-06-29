# FreeText MCQ Evaluation Framework

A comprehensive and extensible framework for evaluating language models on multiple-choice question benchmarks, with a focus on MMLU (Massive Multitask Language Understanding).

## Features

- **Extensible Model Support**: Easy to add new models (OpenAI, Anthropic, local models, etc.)
- **Multiple Benchmark Support**: Currently supports MMLU, easily extensible to other benchmarks
- **Flexible Grading Algorithms**: Start with log-probability grading, easily swap in custom algorithms
- **Comprehensive Evaluation**: Detailed metrics, subject-wise analysis, and confidence scoring
- **Command-line Interface**: Easy-to-use CLI for running evaluations
- **Rich Output**: JSON, CSV, and summary reports

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd freetext-mcq-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Evaluation

#### Using the CLI:
```bash
# Evaluate GPT-4o-mini on MMLU (first 100 questions)
python src/cli.py evaluate --model gpt-4o-mini --max-questions 100

# Evaluate specific subjects
python src/cli.py evaluate --subjects high_school_mathematics computer_security

# List available subjects
python src/cli.py list-subjects

# Get subject statistics
python src/cli.py subject-stats
```

#### Using the example script:
```bash
# Run the example script
python examples/test_gpt4o_mini.py
```

## Framework Architecture

The framework is designed with modularity and extensibility in mind:

```
src/
├── models/           # Model implementations
│   ├── base.py      # Base model interface
│   └── openai_model.py  # OpenAI model implementation
├── benchmarks/       # Benchmark implementations
│   ├── base.py      # Base benchmark interface
│   └── mmlu.py      # MMLU benchmark implementation
├── graders/         # Grading algorithms
│   ├── base.py      # Base grader interface
│   └── logprob_grader.py  # Log probability grader
├── evaluator.py     # Main evaluation engine
└── cli.py          # Command-line interface
```

### Key Components

1. **Models** (`src/models/`): Implement the `BaseModel` interface
2. **Benchmarks** (`src/benchmarks/`): Implement the `BaseBenchmark` interface
3. **Graders** (`src/graders/`): Implement the `BaseGrader` interface
4. **Evaluator**: Orchestrates the evaluation process

## Extending the Framework

### Adding a New Model

Create a new model class that inherits from `BaseModel`:

```python
from src.models.base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        # Initialize your model
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement text generation
        pass
    
    def logprobs(self, prompt: str, completion: str) -> List[float]:
        # Implement log probability calculation
        pass
    
    def score(self, prompt: str, choices: List[str]) -> List[float]:
        # Implement choice scoring
        pass
```

### Adding a New Benchmark

Create a new benchmark class that inherits from `BaseBenchmark`:

```python
from src.benchmarks.base import BaseBenchmark

class MyCustomBenchmark(BaseBenchmark):
    def __init__(self, benchmark_name: str, **kwargs):
        super().__init__(benchmark_name, **kwargs)
    
    def load_data(self) -> None:
        # Load your benchmark data
        pass
    
    def get_questions(self) -> Iterator[Dict[str, Any]]:
        # Yield question data
        pass
    
    def format_question(self, question_data: Dict[str, Any]) -> str:
        # Format question for model input
        pass
    
    def get_choices(self, question_data: Dict[str, Any]) -> List[str]:
        # Return choices
        pass
    
    def get_correct_answer(self, question_data: Dict[str, Any]) -> int:
        # Return correct answer index
        pass
```

### Adding a New Grading Algorithm

Create a new grader class that inherits from `BaseGrader`:

```python
from src.graders.base import BaseGrader

class MyCustomGrader(BaseGrader):
    def __init__(self, grader_name: str, **kwargs):
        super().__init__(grader_name, **kwargs)
    
    def grade_question(self, model_scores: List[float], correct_answer: int) -> Dict[str, Any]:
        # Implement your grading logic
        pass
    
    def grade_batch(self, model_scores_batch: List[List[float]], correct_answers: List[int]) -> List[Dict[str, Any]]:
        # Implement batch grading
        pass
```

## Configuration

The framework uses `config.yaml` for default configurations. You can modify this file to change default settings for models, benchmarks, and evaluation parameters.

## Output Format

The framework generates three types of output files:

1. **Detailed Results** (`results_*.json`): Complete evaluation results for each question
2. **Summary** (`summary_*.json`): Aggregated metrics and statistics
3. **CSV** (`results_*.csv`): Tabular format for easy analysis

### Example Summary Output

```json
{
  "accuracy": 0.75,
  "total_questions": 100,
  "correct_answers": 75,
  "evaluation_time": 120.5,
  "questions_per_second": 0.83,
  "subject_accuracy": {
    "high_school_mathematics": 0.80,
    "computer_security": 0.70
  },
  "avg_confidence": 2.5,
  "model": "gpt-4o-mini",
  "benchmark": "mmlu",
  "grader": "logprob"
}
```

## Supported Models

- **OpenAI Models**: GPT-4o-mini, GPT-4, GPT-3.5-turbo, etc.
- **Extensible**: Easy to add Anthropic, local models, or other APIs

## Supported Benchmarks

- **MMLU**: Massive Multitask Language Understanding (57 subjects)
- **Extensible**: Easy to add other HuggingFace benchmarks

## Grading Algorithms

- **Log Probability**: Standard log-probability based scoring
- **Extensible**: Easy to implement custom grading algorithms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{freetext_mcq_eval,
  title={FreeText MCQ Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/freetext-mcq-eval}
}
``` 