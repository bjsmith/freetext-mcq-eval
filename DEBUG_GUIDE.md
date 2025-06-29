# VSCode Debugging Guide for GPT-4o MMLU Evaluation

This guide explains how to debug the evaluation scripts using VSCode's debugging features.

## Quick Start

1. **Open the project in VSCode**
2. **Set up your environment:**
   ```bash
   python setup.py
   ```
3. **Open the Debug panel** (Ctrl+Shift+D or Cmd+Shift+D)
4. **Select a debug configuration** from the dropdown
5. **Press F5** to start debugging

## Debug Configurations

### 1. Debug Main Evaluation Script
- **Purpose**: Debug the full evaluation script
- **Tasks**: MMLU History only (for faster debugging)
- **Features**: Full argument parsing, error handling
- **Best for**: Testing the complete workflow

### 2. Debug Simple Evaluation
- **Purpose**: Debug the simplified evaluation
- **Tasks**: Both history and geography
- **Features**: Minimal configuration, direct lm-harness usage
- **Best for**: Understanding core lm-harness functionality

### 3. Debug Both MMLU Tasks
- **Purpose**: Debug evaluation on both tasks
- **Tasks**: History and geography
- **Features**: Full evaluation with both tasks
- **Best for**: Complete evaluation testing

### 4. Debug with Custom API Key
- **Purpose**: Debug with explicit API key handling
- **Tasks**: MMLU History
- **Features**: Explicit API key configuration
- **Best for**: Testing API key issues

### 5. Debug Evaluation (Step-by-Step) ⭐ **RECOMMENDED**
- **Purpose**: Step-through debugging with breakpoints
- **Tasks**: MMLU History only
- **Features**: 
  - Built-in breakpoints
  - Connection testing
  - Detailed logging
  - Limited examples (10) for faster debugging
- **Best for**: Learning and troubleshooting

### 6. Debug Evaluation (No Limits)
- **Purpose**: Full evaluation with debugging features
- **Tasks**: Both history and geography
- **Features**: All debugging features without example limits
- **Best for**: Full evaluation with debugging

## Setting Breakpoints

### Automatic Breakpoints
The `debug_evaluation.py` script includes built-in breakpoints at key locations:

1. **Model Setup Complete** (line ~60)
   - After OpenAI model is initialized
   - Good place to inspect model configuration

2. **Before Evaluation Starts** (line ~85)
   - After tasks are loaded
   - Before lm-eval evaluation begins

3. **After Evaluation Completes** (line ~105)
   - After results are obtained
   - Before results are processed

4. **Before Running Evaluation** (line ~180)
   - After model setup and connection testing
   - Before the main evaluation loop

### Manual Breakpoints
You can add breakpoints anywhere in the code by:

1. **Clicking in the gutter** (left margin) next to line numbers
2. **Using F9** to toggle breakpoints
3. **Setting conditional breakpoints** by right-clicking on breakpoints

## Debugging Features

### 1. Variable Inspection
- **Variables panel**: View all local and global variables
- **Watch panel**: Monitor specific variables
- **Debug console**: Execute Python code in the current context

### 2. Call Stack
- **Call Stack panel**: See the execution path
- **Step into** (F11): Go into function calls
- **Step over** (F10): Execute current line and move to next
- **Step out** (Shift+F11): Complete current function and pause at caller

### 3. Logging
The debug script creates detailed logs:
- **Console output**: Real-time logging
- **debug.log file**: Persistent log file
- **Log levels**: DEBUG, INFO, WARNING, ERROR

## Common Debugging Scenarios

### 1. API Key Issues
```bash
# Check if API key is set
python -c "import os; import dotenv; dotenv.load_dotenv(); print('API Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

**Debug steps:**
1. Use "Debug with Custom API Key" configuration
2. Set breakpoint in `setup_openai_model` function
3. Inspect `api_key` variable

### 2. Model Connection Issues
```bash
# Test model connection
python debug_evaluation.py --test-connection
```

**Debug steps:**
1. Use "Debug Evaluation (Step-by-Step)" configuration
2. Set breakpoint in `test_model_connection` function
3. Inspect API responses

### 3. Task Loading Issues
**Debug steps:**
1. Set breakpoint in `get_task_dict` call
2. Inspect `task_dict` variable
3. Check if tasks are properly loaded

### 4. Evaluation Performance Issues
**Debug steps:**
1. Use limited examples (default in debug script)
2. Monitor API call timing
3. Check for rate limiting

## Debug Console Commands

While debugging, you can use these commands in the Debug Console:

```python
# Inspect model
model.model_name
model.api_key[:10] + "..."  # Show first 10 chars of API key

# Inspect tasks
list(task_dict.keys())
task_dict['mmlu_history'].VERSION

# Test model generation
model.generate("Test prompt", max_tokens=10)

# Inspect results
results.keys()
results['results']['mmlu_history']
```

## Troubleshooting

### 1. "Module not found" errors
- Ensure virtual environment is activated
- Check `PYTHONPATH` in launch configuration
- Run `pip install -r requirements.txt`

### 2. API rate limiting
- Increase `max_retries` in model configuration
- Add delays between requests
- Check OpenAI API usage dashboard

### 3. Memory issues
- Reduce `batch_size` in model configuration
- Limit number of examples with `limit` parameter
- Monitor system resources

### 4. Timeout issues
- Increase `timeout` in model configuration
- Check network connectivity
- Reduce `max_new_tokens` if needed

## Tips for Effective Debugging

1. **Start with step-by-step configuration** for learning
2. **Use limited examples** initially for faster iteration
3. **Check logs** in both console and `debug.log` file
4. **Set conditional breakpoints** for specific conditions
5. **Use watch expressions** for frequently accessed variables
6. **Test API connection** before full evaluation
7. **Monitor API costs** during debugging sessions

## Environment Variables

Make sure these are set in your `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

Or set them in the launch configuration environment section.

## Next Steps

After debugging:
1. **Remove breakpoints** for production runs
2. **Increase example limits** for full evaluation
3. **Optimize configuration** based on findings
4. **Save working configurations** for future use 