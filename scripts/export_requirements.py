# scripts/export_requirements.py

import subprocess
import sys
from pathlib import Path

def get_imports(file_path):
    """Get all imports from a Python file"""
    with open(file_path) as f:
        return [
            line.split()[1] 
            for line in f 
            if line.startswith(('import ', 'from ')) 
            and 'amazon_reviews' not in line
        ]

def get_conda_packages():
    """Get list of installed conda packages"""
    result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running 'conda list'")
        sys.exit(1)
    
    packages = {}
    for line in result.stdout.split('\n')[3:]:   
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                packages[parts[0]] = parts[1]
    return packages

def main():
    file_path = Path("src/amazon_reviews/summariser.py")
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        sys.exit(1)

    imports = get_imports(file_path)
    conda_packages = get_conda_packages()
    
    requirements = []
    for imp in imports:
        if imp in conda_packages:
            requirements.append(f"{imp}=={conda_packages[imp]}")
    
    output_path = Path("requirements/summariser.txt")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("\n".join(sorted(requirements)))
    
    print(f"Requirements written to {output_path}")

if __name__ == "__main__":
    main()