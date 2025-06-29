# GPT-4o MMLU History/Geography Evaluation

This project evaluates GPT-4o on MMLU (Massive Multitask Language Understanding) history and geography tasks using the lm-harness framework.

## Setup

### Option 1: Fresh Setup (Recommended)
1. **Install dependencies and create virtual environment:**
   ```bash
   python setup.py
   ```

### Option 2: Migrate from Conda/Base
If you previously set up with conda or base Python distribution:
```bash
python migrate_to_venv.py
```

### Option 3: Manual Setup
1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key:**
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
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run evaluation
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

### Using Virtual Environment Directly
You can also run scripts directly with the virtual environment Python:
```bash
# Without activating venv
./venv/bin/python evaluate_gpt4o_mmlu.py
./venv/bin/python simple_evaluation.py
./venv/bin/python debug_evaluation.py
```

### Command Line Arguments
- `--api-key`: OpenAI API key (optional if set in environment)
- `--output`: Output file for results (default: `gpt4o_mmlu_results.json`)
- `--tasks`: Tasks to evaluate (choices: `mmlu_history`, `mmlu_geography`)

## VSCode Debugging

The project is configured for VSCode debugging with virtual environment support:

1. **Open the project in VSCode**
2. **Open the Debug panel** (Ctrl+Shift+D or Cmd+Shift+D)
3. **Select a debug configuration** from the dropdown
4. **Press F5** to start debugging

### Debug Configurations Available:
- **Debug Evaluation (Step-by-Step)** ⭐ - Best for learning
- **Debug Main Evaluation Script** - Full workflow
- **Debug Simple Evaluation** - Core functionality
- **Debug Both MMLU Tasks** - Complete evaluation
- **Debug with Custom API Key** - API troubleshooting

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

## Virtual Environment

This project uses a Python virtual environment (`venv`) for dependency isolation:

- **Location**: `./venv/`
- **Python**: `./venv/bin/python` (Unix/Mac) or `./venv/Scripts/python.exe` (Windows)
- **Pip**: `./venv/bin/pip` (Unix/Mac) or `./venv/Scripts/pip.exe` (Windows)

### Activating the Virtual Environment:
```bash
# Unix/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Deactivating:
```bash
deactivate
```

## Requirements

- Python 3.8+
- OpenAI API access
- Internet connection for API calls

## Notes

- The evaluation uses temperature=0 for deterministic results
- No few-shot examples are used (num_fewshot=0)
- Bootstrap confidence intervals are calculated with 1000 iterations
- Results are saved in JSON format for further analysis
- All dependencies are isolated in the virtual environment

## Troubleshooting

### Virtual Environment Issues
- **"No module named 'lm_eval'"**: Make sure you're using the virtual environment Python
- **"Permission denied"**: Make sure the venv directory has proper permissions
- **"Command not found"**: Use the full path to venv Python: `./venv/bin/python`

### VSCode Issues
- **Wrong Python interpreter**: Select the venv Python in VSCode (Ctrl+Shift+P → "Python: Select Interpreter")
- **Debug not working**: Check that launch.json points to `./venv/bin/python`

### API Issues
- **Rate limiting**: Increase `max_retries` in model configuration
- **Authentication errors**: Check your `.env` file and API key 