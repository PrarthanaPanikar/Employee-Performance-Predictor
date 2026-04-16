# Employee Performance Predictor - Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Processor**: Dual-core processor or better

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **Processor**: Quad-core processor or better

---

## Installation Instructions by Platform

### Windows Installation

#### Step 1: Install Python
1. Download Python from [python.org](https://python.org)
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify installation:
```cmd
python --version
pip --version
```

#### Step 2: Create Project Directory
```cmd
mkdir "Employee Performance Predictor"
cd "Employee Performance Predictor"
```

#### Step 3: Create Virtual Environment
```cmd
python -m venv venv
```

#### Step 4: Activate Virtual Environment
```cmd
venv\Scripts\activate
```

#### Step 5: Upgrade Pip
```cmd
python -m pip install --upgrade pip
```

#### Step 6: Install Dependencies
```cmd
pip install -r requirements.txt
```

#### Step 7: Verify Installation
```cmd
python -c "import pandas, sklearn, streamlit; print('Installation successful!')"
```

---

### macOS Installation

#### Step 1: Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install Python
```bash
brew install python@3.9
```

#### Step 3: Create Project Directory
```bash
mkdir "Employee Performance Predictor"
cd "Employee Performance Predictor"
```

#### Step 4: Create Virtual Environment
```bash
python3 -m venv venv
```

#### Step 5: Activate Virtual Environment
```bash
source venv/bin/activate
```

#### Step 6: Upgrade Pip
```bash
python -m pip install --upgrade pip
```

#### Step 7: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 8: Verify Installation
```bash
python -c "import pandas, sklearn, streamlit; print('Installation successful!')"
```

---

### Linux Installation (Ubuntu/Debian)

#### Step 1: Update System Packages
```bash
sudo apt update && sudo apt upgrade -y
```

#### Step 2: Install Python and Pip
```bash
sudo apt install python3 python3-pip python3-venv -y
```

#### Step 3: Install Additional Dependencies
```bash
sudo apt install build-essential python3-dev -y
```

#### Step 4: Create Project Directory
```bash
mkdir "Employee Performance Predictor"
cd "Employee Performance Predictor"
```

#### Step 5: Create Virtual Environment
```bash
python3 -m venv venv
```

#### Step 6: Activate Virtual Environment
```bash
source venv/bin/activate
```

#### Step 7: Upgrade Pip
```bash
python -m pip install --upgrade pip
```

#### Step 8: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 9: Verify Installation
```bash
python -c "import pandas, sklearn, streamlit; print('Installation successful!')"
```

---

## Requirements.txt File

Create a file named `requirements.txt` with the following content:

```txt
# Core Data Science Libraries
pandas==1.5.3
numpy==1.24.3
scipy==1.10.1

# Machine Learning
scikit-learn==1.2.2
xgboost==1.7.5
joblib==1.2.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# Web Application
streamlit==1.25.0

# Jupyter Notebooks
jupyter==1.0.0
notebook==6.5.4

# Utilities
pyyaml==6.0
tqdm==4.64.1
```

---

## Alternative Installation Methods

### Using Conda (Recommended for Data Scientists)

#### Step 1: Install Anaconda or Miniconda
Download from [anaconda.com](https://anaconda.com) or use Miniconda for minimal installation.

#### Step 2: Create Conda Environment
```bash
conda create -n emp_perf python=3.9 -y
```

#### Step 3: Activate Environment
```bash
conda activate emp_perf
```

#### Step 4: Install Packages
```bash
conda install pandas numpy scikit-learn matplotlib seaborn -y
pip install streamlit xgboost plotly jupyter
```

---

### Using Docker (Advanced Users)

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py"]
```

#### Step 2: Build Docker Image
```bash
docker build -t employee-performance-predictor .
```

#### Step 3: Run Docker Container
```bash
docker run -p 8501:8501 employee-performance-predictor
```

---

## IDE Setup

### Visual Studio Code (Recommended)

#### Step 1: Install VS Code
Download from [code.visualstudio.com](https://code.visualstudio.com)

#### Step 2: Install Extensions
- Python extension by Microsoft
- Jupyter extension by Microsoft
- Python Docstring Generator
- GitLens

#### Step 3: Configure VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "jupyter.jupyterServerType": "local"
}
```

### PyCharm Setup

#### Step 1: Configure Python Interpreter
1. Open PyCharm
2. Go to File > Settings > Project > Python Interpreter
3. Click "Add Interpreter" > "Existing Environment"
4. Select your virtual environment's Python executable

#### Step 2: Configure Project Structure
1. Right-click on folders and mark as:
   - `src/` as "Source Root"
   - `tests/` as "Test Sources Root"

---

## Verification Steps

### Step 1: Test Python Installation
```python
# test_python.py
import sys
print(f"Python version: {sys.version}")

import pandas as pd
print(f"Pandas version: {pd.__version__}")

import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")

import streamlit as st
print(f"Streamlit version: {st.__version__}")

print("All packages installed successfully!")
```

Run the test:
```bash
python test_python.py
```

### Step 2: Test Jupyter Notebook
```bash
jupyter notebook
```
Create a new notebook and test basic imports:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
print("Jupyter environment working!")
```

### Step 3: Test Streamlit App
Create a simple test app `test_app.py`:
```python
import streamlit as st
import pandas as pd
import numpy as np

st.title("Installation Test")
st.write("If you can see this, Streamlit is working!")

# Test data manipulation
df = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10)
})
st.write("Sample DataFrame:")
st.dataframe(df)

st.success("Installation successful!")
```

Run the test app:
```bash
streamlit run test_app.py
```

---

## Common Installation Issues and Solutions

### Issue 1: Python not found in PATH
**Windows Solution:**
1. Reinstall Python and check "Add Python to PATH"
2. Or manually add Python to PATH:
   - Find Python installation directory
   - Add to System Environment Variables

**macOS/Linux Solution:**
```bash
which python3
export PATH="/path/to/python:$PATH"
echo 'export PATH="/path/to/python:$PATH"' >> ~/.bashrc
```

### Issue 2: Virtual Environment Activation Fails
**Windows:**
```cmd
# Try full path
C:\path\to\project\venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
# Check if venv was created correctly
ls -la venv/bin/
# Recreate if necessary
rm -rf venv
python3 -m venv venv
```

### Issue 3: Package Installation Fails
**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages individually to identify problematic one
pip install pandas
pip install numpy
pip install scikit-learn

# Use --no-cache-dir flag
pip install --no-cache-dir package_name
```

### Issue 4: Permission Errors
**macOS/Linux:**
```bash
# Use user directory
pip install --user package_name

# Or fix permissions
sudo chown -R $USER ~/.local
```

**Windows:**
```cmd
# Run as administrator
# Or install in user directory
pip install --user package_name
```

### Issue 5: Jupyter Notebook Kernel Issues
**Solution:**
```bash
# Install ipykernel
pip install ipykernel

# Register virtual environment
python -m ipykernel install --user --name=emp_perf

# Restart Jupyter and select correct kernel
```

---

## Environment Variables (Optional)

Create a `.env` file for environment-specific settings:
```env
# Project Configuration
PROJECT_NAME=Employee Performance Predictor
VERSION=1.0.0
DEBUG=False

# Data Paths
DATA_PATH=./data
MODEL_PATH=./models
OUTPUT_PATH=./outputs

# Model Parameters
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5
```

Load environment variables in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv('PROJECT_NAME')
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
```

---

## Next Steps After Installation

1. **Verify Installation**: Run all verification steps
2. **Explore Project Structure**: Review folder organization
3. **Run Data Generation**: Create synthetic dataset
4. **Start EDA**: Begin exploratory data analysis
5. **Train Models**: Build machine learning pipeline
6. **Launch Dashboard**: Run Streamlit application

---

## Getting Help

If you encounter installation issues:

1. **Check Python Version**: Ensure you have Python 3.8+
2. **Verify Virtual Environment**: Make sure it's activated
3. **Update Packages**: Use latest versions of dependencies
4. **Check Internet Connection**: Some packages require downloads
5. **Consult Documentation**: Refer to official package documentation

### Useful Commands
```bash
# Check Python version
python --version

# List installed packages
pip list

# Check package version
pip show pandas

# Upgrade specific package
pip install --upgrade pandas

# Uninstall package
pip uninstall package_name

# Reinstall package
pip install --force-reinstall package_name
```

This installation guide should help you set up the Employee Performance Predictor project on any platform. Follow the steps carefully and verify each installation before proceeding to the next step.
