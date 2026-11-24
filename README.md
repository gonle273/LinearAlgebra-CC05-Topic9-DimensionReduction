# Topic 1.9: Dimension Reduction for Clustering - PCA and LDA

Linear Algebra with Machine Learning and Data (Crista Arangala, 2023)  
**Group 9 Project Repository**

## 1. Project Overview
This repository contains the numerical implementations and experimental reproduction for Topic 1.9, focusing on:
- Principal Component Analysis (PCA) via Truncated SVD.
- Linear Discriminant Analysis (LDA) via generalized eigenvalue problems.
- Nearest-Mean classification comparison on dimensionality-reduced subspaces.

## 2. Directory Structure
- `src/`: Core reusable algorithms implemented from scratch in pure NumPy (`pca.py`, `lda.py`, `classifier.py`, `data.py`, `plotting.py`).
- `experiments/`: Scripts for executing experimental setups (e.g., `c3_digits_pca_vs_lda.py`).
- `outputs/`: Automatically generated publication-ready figures (`.png`, `.pdf`) and LaTeX tables (`.tex`, `.csv`).
- `run_all.py`: Master pipeline execution script.

## 3. Getting Started
Clone the repository and install the minimal dependencies:
```bash
git clone https://github.com/gonle273/LinearAlgebra-CC05-Topic9-DimensionReduction.git
cd LinearAlgebra-CC05-Topic9-DimensionReduction
pip install -r requirements.txt
