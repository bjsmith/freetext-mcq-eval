# GPT-4o MMLU History/Geography Evaluation

This project evaluates GPT-4o on MMLU (Massive Multitask Language Understanding) history and geography tasks using the lm-harness framework.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key:**
   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```
   
   Or set the environment variable directly:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

## Usage

### Basic Usage
Run the evaluation on both history and geography tasks:
```bash
python evaluate_gpt4o_mmlu.py
```

### Custom Options
```bash
# Evaluate only history
python evaluate_gpt4o_mmlu.py --tasks mmlu_history

# Evaluate only geography
python evaluate_gpt4o_mmlu.py --tasks mmlu_geography

# Specify custom output file
python evaluate_gpt4o_mmlu.py --output my_results.json

# Provide API key directly
python evaluate_gpt4o_mmlu.py --api-key "your_api_key_here"
```

### Command Line Arguments
- `--api-key`: OpenAI API key (optional if set in environment)
- `--output`: Output file for results (default: `gpt4o_mmlu_results.json`)
- `--tasks`: Tasks to evaluate (choices: `mmlu_history`, `mmlu_geography`)

## Output

The script generates a JSON file with detailed evaluation results including:
- Accuracy scores for each task
- Normalized accuracy scores
- Number of examples evaluated
- Detailed metrics

## MMLU Tasks

### History
- Ancient civilizations
- World wars
- Political history
- Cultural history

### Geography
- Physical geography
- Human geography
- Economic geography
- Political geography

## Configuration

The evaluation settings can be modified in `mmlu_config.yaml`:
- Model parameters (temperature, max tokens, etc.)
- Evaluation settings (bootstrap iterations, random seed)
- Output format and file location

## Requirements

- Python 3.8+
- OpenAI API access
- Internet connection for API calls

## Notes

- The evaluation uses temperature=0 for deterministic results
- No few-shot examples are used (num_fewshot=0)
- Bootstrap confidence intervals are calculated with 1000 iterations
- Results are saved in JSON format for further analysis 