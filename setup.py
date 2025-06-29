#!/usr/bin/env python3
"""
Setup script for GPT-4o MMLU evaluation project.
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
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
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import lm_eval
        import openai
        import dotenv
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    print("Setting up GPT-4o MMLU evaluation project...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
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
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nYou can now run the evaluation:")
    print("  python evaluate_gpt4o_mmlu.py")
    print("  python simple_evaluation.py")
    
    return 0

if __name__ == "__main__":
    exit(main()) 