"""
Experiment C3: Dimension Reduction Comparison (PCA vs. LDA) on Handwritten Digits.
Evaluates 2D projections using Nearest-Mean Classifier.
"""

import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.pca import MyPCA
from src.lda import MyLDA
from src.classifier import NearestMeanClassifier
from src.data import load_digit_classes, train_test_split_custom
from src.plotting import plot_side_by_side_2d


def run_experiment_c3():
    print("[INFO] Running Experiment C3: PCA vs. LDA Comparison...")

    # 1. Load dataset (Classes 0, 1, 2) and split train/test
    classes = [0, 1, 2]
    X, y = load_digit_classes(classes=classes)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, train_ratio=0.8, random_state=42)

    # 2. Evaluate in Raw Space (64-D)
    clf_raw = NearestMeanClassifier().fit(X_train, y_train)
    acc_raw = clf_raw.score(X_test, y_test)

    # 3. Fit and Transform PCA (2D)
    pca = MyPCA(n_components=2).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    clf_pca = NearestMeanClassifier().fit(X_train_pca, y_train)
    acc_pca = clf_pca.score(X_test_pca, y_test)

    # 4. Fit and Transform LDA (2D)
    lda = MyLDA(n_components=2).fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    clf_lda = NearestMeanClassifier().fit(X_train_lda, y_train)
    acc_lda = clf_lda.score(X_test_lda, y_test)

    # 5. Print Summary Table
    print("\n" + "=" * 50)
    print("      NEAREST-MEAN CLASSIFIER ACCURACY")
    print("=" * 50)
    print(f" Raw Digits (64-D) : {acc_raw * 100:.2f}%")
    print(f" PCA Space (2-D)   : {acc_pca * 100:.2f}%")
    print(f" LDA Space (2-D)   : {acc_lda * 100:.2f}%")
    print("=" * 50)

    # 6. Save Numerical Results to outputs/tables/ (CSV and LaTeX format)
    table_csv_path = os.path.join(PROJECT_ROOT, "outputs", "tables", "c3_results.csv")
    table_tex_path = os.path.join(PROJECT_ROOT, "outputs", "tables", "c3_results.tex")
    
    # Save CSV
    with open(table_csv_path, "w", encoding="utf-8") as f:
        f.write("Representation,Dimensions,Accuracy\n")
        f.write(f"Raw Features,{X.shape[1]},{acc_raw:.4f}\n")
        f.write(f"PCA Projection,2,{acc_pca:.4f}\n")
        f.write(f"LDA Projection,2,{acc_lda:.4f}\n")

    # Save LaTeX Table for Friend E
    with open(table_tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Classification Accuracy of Nearest-Mean Classifier on Digits (0, 1, 2)}\n")
        f.write("\\label{tab:c3_results}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Representation Space} & \\textbf{Dimension} & \\textbf{Test Accuracy (\\%)} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Original Raw Features & {X.shape[1]} & {acc_raw * 100:.2f}\\% \\\\\n")
        f.write(f"PCA Subspace (Unsupervised) & 2 & {acc_pca * 100:.2f}\\% \\\\\n")
        f.write(f"LDA Subspace (Supervised) & 2 & \\textbf{{{acc_lda * 100:.2f}\\%}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[OUTPUT] Tables generated:\n  -> {table_csv_path}\n  -> {table_tex_path}")


if __name__ == "__main__":
    run_experiment_c3()