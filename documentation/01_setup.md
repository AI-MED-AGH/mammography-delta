## First Time Setup

### Prerequisites

* **Python 3.9** is required (newer versions may cause conflicts with PyRadiomics).
* Ensure Python 3.9 is added to your system PATH.

### Installation Steps

1. **Create a Virtual Environment**

```bash
# Linux/macOS
python3.9 -m venv venv
```

```bash
# Windows (try 'python' if 'python3.9' is not recognized)
python3.9 -m venv venv
```

2. **Activate the Environment**

```bash
# Linux/macOS
source venv/bin/activate
```

```bash
# Windows (Command Prompt / PowerShell)
.\venv\Scripts\activate
```

3. **Install Dependencies**
   *Note: The installation order is critical. NumPy must be installed before building PyRadiomics.*

```bash
# 1. Upgrade installer tools
pip install --upgrade pip setuptools wheel

# 2. Install specific NumPy version first (crucial for PyRadiomics build)
pip install numpy==1.26.4

# 3. Install PyRadiomics preventing build isolation
pip install pyradiomics==3.0.1 --no-build-isolation

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Verification

To verify the installation was successful, run:

```bash
python -c "import radiomics; print(f'PyRadiomics {radiomics.__version__} installed successfully!')"
```

---