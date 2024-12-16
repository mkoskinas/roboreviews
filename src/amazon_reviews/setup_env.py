import subprocess
import sys
from pathlib import Path

def setup_environment(model_type: str):
    """Setup specific environment for each model type"""
    project_root = Path(__file__).parent.parent.parent.parent
    requirements_dir = project_root / "requirements"
    
    if model_type not in ["classifier", "clusterer", "summarizer"]:
        raise ValueError(f"Unknown model type: {model_type}")
    
    req_file = requirements_dir / f"{model_type}.txt"
    
    print(f"Setting up environment for {model_type}...")
    try:
        # Create new virtual environment
        venv_path = project_root / f".venv_{model_type}"
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        
        # Install requirements
        pip_path = venv_path / "bin" / "pip"
        subprocess.check_call([str(pip_path), "install", "-r", str(req_file)])
        
        print(f"Environment setup complete. Activate with:")
        print(f"source {venv_path}/bin/activate")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up environment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_env.py [classifier|clusterer|summarizer]")
        sys.exit(1)
    
    setup_environment(sys.argv[1])