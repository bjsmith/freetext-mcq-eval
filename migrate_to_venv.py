#!/usr/bin/env python3
"""
Migration script to transition from conda/base to virtual environment.
"""

import os
import shutil
import subprocess
import sys

def check_current_environment():
    """Check what Python environment is currently being used."""
    print("Checking current Python environment...")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    if conda_env:
        print(f"Currently in conda environment: {conda_env}")
        return "conda"
    else:
        print("Not in a conda environment")
        return "system"

def remove_existing_venv():
    """Remove existing venv directory if it exists."""
    venv_path = "venv"
    if os.path.exists(venv_path):
        print(f"Removing existing virtual environment at {venv_path}...")
        try:
            shutil.rmtree(venv_path)
            print("✓ Existing virtual environment removed")
            return True
        except Exception as e:
            print(f"✗ Failed to remove existing venv: {e}")
            return False
    else:
        print("✓ No existing virtual environment found")
        return True

def create_fresh_venv():
    """Create a fresh virtual environment."""
    print("Creating fresh virtual environment...")
    try:
        import venv
        venv.create("venv", with_pip=True)
        print("✓ Fresh virtual environment created")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def install_dependencies_in_venv():
    """Install dependencies in the virtual environment."""
    print("Installing dependencies in virtual environment...")
    try:
        if sys.platform == "win32":
            pip_path = os.path.join("venv", "Scripts", "pip.exe")
        else:
            pip_path = os.path.join("venv", "bin", "pip")
        
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed in virtual environment")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def test_venv_setup():
    """Test that the virtual environment is working correctly."""
    print("Testing virtual environment setup...")
    try:
        if sys.platform == "win32":
            python_path = os.path.join("venv", "Scripts", "python.exe")
        else:
            python_path = os.path.join("venv", "bin", "python")
        
        test_script = """
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import lm_eval
    print("✓ lm-eval imported successfully")
except ImportError as e:
    print(f"✗ Failed to import lm-eval: {e}")

try:
    import openai
    print("✓ openai imported successfully")
except ImportError as e:
    print(f"✗ Failed to import openai: {e}")

try:
    import dotenv
    print("✓ python-dotenv imported successfully")
except ImportError as e:
    print(f"✗ Failed to import dotenv: {e}")
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"✗ Test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to test venv setup: {e}")
        return False

def update_vscode_config():
    """Update VSCode configuration to use the virtual environment."""
    print("Updating VSCode configuration...")
    
    # Update settings.json
    settings_file = ".vscode/settings.json"
    if os.path.exists(settings_file):
        try:
            import json
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            # Update Python interpreter path
            if sys.platform == "win32":
                settings["python.defaultInterpreterPath"] = "./venv/Scripts/python.exe"
            else:
                settings["python.defaultInterpreterPath"] = "./venv/bin/python"
            
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            
            print("✓ VSCode settings updated")
        except Exception as e:
            print(f"✗ Failed to update VSCode settings: {e}")
    
    # Update launch.json
    launch_file = ".vscode/launch.json"
    if os.path.exists(launch_file):
        try:
            import json
            with open(launch_file, 'r') as f:
                launch_config = json.load(f)
            
            # Update Python path in all configurations
            python_path = "./venv/bin/python" if sys.platform != "win32" else "./venv/Scripts/python.exe"
            for config in launch_config.get("configurations", []):
                config["python"] = python_path
            
            with open(launch_file, 'w') as f:
                json.dump(launch_config, f, indent=4)
            
            print("✓ VSCode launch configuration updated")
        except Exception as e:
            print(f"✗ Failed to update VSCode launch config: {e}")

def main():
    print("Migrating from conda/base to virtual environment...")
    print("=" * 60)
    
    # Check current environment
    current_env = check_current_environment()
    
    # Remove existing venv
    if not remove_existing_venv():
        return 1
    
    # Create fresh venv
    if not create_fresh_venv():
        return 1
    
    # Install dependencies
    if not install_dependencies_in_venv():
        return 1
    
    # Test setup
    if not test_venv_setup():
        return 1
    
    # Update VSCode config
    update_vscode_config()
    
    print("\n" + "=" * 60)
    print("Migration completed successfully!")
    print("\nVirtual environment is now ready at: ./venv")
    
    if sys.platform == "win32":
        python_path = "./venv/Scripts/python.exe"
        activate_cmd = "venv\\Scripts\\activate"
    else:
        python_path = "./venv/bin/python"
        activate_cmd = "source venv/bin/activate"
    
    print(f"\nTo activate the virtual environment:")
    print(f"  {activate_cmd}")
    
    print(f"\nTo run scripts with the virtual environment:")
    print(f"  {python_path} evaluate_gpt4o_mmlu.py")
    print(f"  {python_path} simple_evaluation.py")
    print(f"  {python_path} debug_evaluation.py")
    
    print(f"\nVSCode is now configured to use the virtual environment.")
    print(f"Open VSCode and start debugging!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 