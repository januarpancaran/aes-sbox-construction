# AES S-box Construction

A comprehensive web-based tool for exploring and constructing AES S-boxes through affine matrix optimization. This project implements advanced cryptographic techniques to analyze and modify the standard AES S-box for increased strength.

## 📋 Project Overview

This application explores the construction of substitution boxes (S-boxes) used in the Advanced Encryption Standard (AES) algorithm. The project focuses on:

- **Affine Matrix Exploration**: Generate and analyze 8×8 affine matrices using circular right shift patterns
- **S-box Construction**: Build custom S-boxes using irreducible polynomials and affine transformations
- **S-box Testing**: Evaluate S-box properties (linearity, differential uniformity, non-linearity)
- **AES Implementation**: Compare standard AES with custom S-box variants
- **Results Comparison**: Analyze and visualize results across different S-box configurations

## 🚀 Features

### 1. Affine Matrix Explorer
- Generate 256 possible 8×8 affine matrices
- Explore matrices using circular right shift patterns
- Visualize matrix structures and properties
- Interactive parameter adjustment

### 2. S-box Constructor
- Compute multiplicative inverses in GF(2⁸)
- Apply affine transformations with custom matrices
- Support for irreducible polynomial: x⁸ + x⁴ + x³ + x + 1
- Optimized caching for fast performance

### 3. S-box Tester
- Test S-box cryptographic properties:
  - Linearity/Balance analysis
  - Differential uniformity
  - Non-linearity evaluation
  - Avalanche effect measurement
- Generate detailed test reports

### 4. Results & Comparison
- Compare multiple S-box configurations
- Analyze cryptographic strength metrics
- Export results for further analysis
- Visual comparison charts

### 5. AES Implementation
- Standard AES encryption/decryption
- Support for custom S-boxes
- Performance metrics
- Example encryption demonstrations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/januarpancaran/aes-sbox-construction.git
cd aes-sbox-construction
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
```

3. **Activate virtual environment:**
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

Key dependencies include:
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Cryptography**: Cryptographic operations

See `requirements.txt` for the complete list.

## 🏃 Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### AES S-box Modification
The project implements techniques based on affine matrix exploration to modify the standard AES S-box:

1. **Multiplicative Inverse**: Computed in Galois Field GF(2⁸) using the irreducible polynomial
2. **Affine Transformation**: Applied using 8×8 matrices with various configurations
3. **Affine Constant**: 8-bit constant added during transformation

### Key Operations
- **GF(2⁸) Multiplication**: Polynomial multiplication with irreducible polynomial
- **Matrix Generation**: Circular right shift pattern from first row
- **S-box Evaluation**: Testing cryptographic strength metrics

## 📊 Usage Examples

### 1. Explore Affine Matrices
Navigate to "Affine Matrix Exploration" to:
- View different 8×8 matrix configurations
- Analyze matrix properties
- Export matrix data

### 2. Construct Custom S-boxes
Use "S-box Construction" to:
- Create S-boxes with custom affine matrices
- Apply different constants
- View transformation mappings

### 3. Test S-boxes
In "S-box Testing" section:
- Run cryptographic property tests
- Analyze linearity and differential uniformity
- Compare metrics with standard AES

### 4. Analyze Results
Use "Results & Comparison" to:
- Compare multiple S-box configurations
- Generate comparison reports
- Export analysis data

## 📈 Metrics & Evaluation

The application evaluates S-boxes using:

- **Non-linearity**: Measures the distance from linear Boolean functions
- **Differential Uniformity**: Evaluates uniform distribution of differences
- **Linearity/Balance**: Checks balanced output distribution
- **Avalanche Effect**: Measures how input changes affect outputs