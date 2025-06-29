#!/usr/bin/env python3
"""
Setup script for GPT-4o MMLU evaluation project.
"""

import os
import subprocess
import sys
import venv

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print("✓ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        venv.create(venv_path, with_pip=True)
        print("✓ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get the path to the virtual environment's Python executable."""
    if sys.platform == "win32":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")

def get_venv_pip():
    """Get the path to the virtual environment's pip executable."""
    if sys.platform == "win32":
        return os.path.join("venv", "Scripts", "pip.exe")
    else:
        return os.path.join("venv", "bin", "pip")

def install_dependencies():
    """Install required dependencies in the virtual environment."""
    print("Installing dependencies in virtual environment...")
    try:
        pip_path = get_venv_pip()
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully in virtual environment")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def setup_env_file():
    """Create .env file if it doesn't exist."""
    env_file = ".env"
    if os.path.exists(env_file):
        print("✓ .env file already exists")
        return True
    
    print("Creating .env file...")
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("✗ API key is required")
        return False
    
    try:
        with open(env_file, "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print("✓ .env file created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported using venv Python."""
    print("Testing imports with virtual environment...")
    try:
        python_path = get_venv_python()
        test_script = """
import sys
import lm_eval
import openai
import dotenv
print("✓ All required packages imported successfully")
"""
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"✗ Import error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to test imports: {e}")
        return False

def update_vscode_settings():
    """Update VSCode settings to use the virtual environment."""
    settings_file = ".vscode/settings.json"
    
    # Get the absolute path to the venv Python
    venv_python = os.path.abspath(get_venv_python())
    
    if os.path.exists(settings_file):
        print("✓ VSCode settings already configured for venv")
        return True
    
    print("Configuring VSCode settings for virtual environment...")
    try:
        os.makedirs(".vscode", exist_ok=True)
        settings_content = {
            "python.defaultInterpreterPath": venv_python,
            "python.terminal.activateEnvironment": True,
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": False,
            "python.linting.flake8Enabled": True,
            "python.formatting.provider": "black",
            "python.sortImports.args": ["--profile", "black"],
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/.DS_Store": True
            },
            "python.analysis.extraPaths": [
                "./venv/lib/python3.*/site-packages"
            ],
            "python.envFile": "${workspaceFolder}/.env",
            "python.terminal.activateEnvInCurrentTerminal": True,
            "debug.console.fontSize": 12,
            "debug.console.lineHeight": 1.2,
            "python.testing.pytestEnabled": False,
            "python.testing.unittestEnabled": False,
            "python.testing.nosetestsEnabled": False,
            "python.testing.pytestArgs": [],
            "python.testing.unittestArgs": [],
            "python.testing.nosetestsArgs": [],
            "python.testing.autoTestDiscoverOnSaveEnabled": False
        }
        
        import json
        with open(settings_file, 'w') as f:
            json.dump(settings_content, f, indent=4)
        
        print("✓ VSCode settings configured for virtual environment")
        return True
    except Exception as e:
        print(f"✗ Failed to configure VSCode settings: {e}")
        return False

def main():
    print("Setting up GPT-4o MMLU evaluation project with virtual environment...")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    if not create_venv():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Test imports
    if not test_imports():
        return 1
    
    # Setup environment file
    if not setup_env_file():
        return 1
    
    # Update VSCode settings
    if not update_vscode_settings():
        return 1
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("\nVirtual environment created at: ./venv")
    print(f"Python interpreter: {os.path.abspath(get_venv_python())}")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    
    print("\nYou can now run the evaluation:")
    print(f"  {get_venv_python()} evaluate_gpt4o_mmlu.py")
    print(f"  {get_venv_python()} simple_evaluation.py")
    print(f"  {get_venv_python()} debug_evaluation.py")
    
    print("\nOr use VSCode debugging (already configured for venv)")
    
    return 0

if __name__ == "__main__":
    exit(main()) 