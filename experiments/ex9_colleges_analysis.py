"""
Exercise 9 Analysis: PCA and LDA on East Coast Colleges and Universities (Table 3.5).
Covers standardization, PCA projection, projections onto custom lines, 
Within-class scatter matrices, and LDA direction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in path to import shared src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.pca import MyPCA
from src.lda import MyLDA


def run_exercise_9():
    print("[INFO] Running Exercise 9: College Data Analysis...")

    # 1. Load Data from Table 3.5 (East Coast Colleges)
    data = {
        "Institution": [
            "Bucknell University", "Davidson College", "Florida State University",
            "Georgia Tech University", "Lafayette College", "University of North Carolina",
            "University of Richmond", "University of Virginia", "Villanova University", "William and Mary University"
        ],
        "Tuition": [56092, 51447, 18786, 33020, 53630, 35170, 52610, 49032, 56730, 45272],
        "Undergraduates": [4900, 1843, 32812, 16049, 2642, 19117, 3227, 11786, 10983, 6377],
        "AcceptanceRate": [0.31, 0.20, 0.50, 0.23, 0.31, 0.24, 0.33, 0.27, 0.29, 0.34],
        "GraduationRate": [0.90, 0.93, 0.80, 0.85, 0.90, 0.90, 0.88, 0.93, 0.91, 0.90]
    }
    
    df = pd.DataFrame(data)
    institutions = df["Institution"].values
    X_raw = df[["Tuition", "Undergraduates", "AcceptanceRate", "GraduationRate"]].values

    # 2. Part (a): Standardization (mean = 0, std = 1 using divisor N-1)
    mean_vals = np.mean(X_raw, axis=0)
    std_vals = np.std(X_raw, axis=0, ddof=1)
    X_std = (X_raw - mean_vals) / std_vals

    # Apply PCA (via SVD) to project into R^2
    pca = MyPCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Align signs for consistency (making largest loading positive if needed)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Institution"] = institutions
    
    print("\n--- PART (A) & (B): PCA Coordinates & Distance to University of Virginia ---")
    va_idx = np.where(df_pca["Institution"] == "University of Virginia")[0][0]
    va_coords = X_pca[va_idx]

    distances = []
    for i, inst in enumerate(institutions):
        dist = np.linalg.norm(X_pca[i] - va_coords)
        distances.append(dist)
    
    df_pca["Distance_to_VA"] = distances
    print(df_pca.sort_values(by="Distance_to_VA")[["Institution", "PC1", "PC2", "Distance_to_VA"]])

    # 3. Part (c): Projection onto line y = 0.25x - 0.5 passing through point (0, -0.5)
    # Line equation: vector direction u = [1, 0.25], base point p0 = [0, -0.5]
    u = np.array([1.0, 0.25])
    u = u / np.linalg.norm(u) # unit vector
    p0 = np.array([0.0, -0.5])

    line_positions = []
    for coords in X_pca:
        # Signed position along the line: dot product with unit vector
        pos = np.dot(coords - p0, u)
        line_positions.append(pos)
    
    df_pca["Line_Position"] = line_positions
    print("\n--- PART (C): Projections onto line y = 0.25x - 0.5 ---")
    va_line_pos = df_pca.loc[df_pca["Institution"] == "University of Virginia", "Line_Position"].values[0]
    df_pca["Dist_on_Line"] = np.abs(df_pca["Line_Position"] - va_line_pos)
    print(df_pca.sort_values(by="Dist_on_Line")[["Institution", "Line_Position", "Dist_on_Line"]])

    # 4. Part (d): Grouping by Undergraduates (< 10,000 vs >= 10,000)
    # Group 1: Undergrads < 10000, Group 2: Undergrads >= 10000
    groups = []
    for ug in df["Undergraduates"]:
        groups.append(1 if ug < 10000 else 2)
    df_pca["Group"] = groups
    y_labels = np.array(groups)

    # Compute S_W and S_B on the PCA coordinates (or standardized data)
    lda = MyLDA(n_components=1)
    # Let's manually compute S_W and S_B to display for the report
    classes = np.unique(y_labels)
    D_pca = X_pca.shape[1]
    overall_mean_pca = np.mean(X_pca, axis=0)
    
    S_W = np.zeros((D_pca, D_pca))
    S_B = np.zeros((D_pca, D_pca))
    
    for c in classes:
        X_c = X_pca[y_labels == c]
        N_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        
        diff = X_c - mean_c
        S_W += np.dot(diff.T, diff)
        
        mean_diff = (mean_c - overall_mean_pca).reshape(-1, 1)
        S_B += N_c * np.dot(mean_diff, mean_diff.T)

    print("\n--- PART (D): Scatter Matrices S_W and S_B ---")
    print("Within-class Scatter Matrix (S_W):\n", np.round(S_W, 4))
    print("Between-class Scatter Matrix (S_B):\n", np.round(S_B, 4))

    # 5. Part (e): LDA Direction
    lda_scores = lda.fit_transform(X_pca, y_labels)
    df_pca["LDA_Score"] = lda_scores.flatten()
    print("\n--- PART (E): LDA Scores & Classification ---")
    print(df_pca.sort_values(by="LDA_Score", ascending=False)[["Institution", "Group", "PC1", "LDA_Score"]])

    # 6. Save summary table
    out_table = os.path.join(PROJECT_ROOT, "outputs", "tables", "ex9_summary.csv")
    df_pca.to_csv(out_table, index=False)
    print(f"[OUTPUT] Exercise 9 summary saved to {out_table}")

    # 7. Generate and Save PCA Scatter Plot for Colleges
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df_pca["PC1"], df_pca["PC2"], c='royalblue', edgecolors='k', s=60)
    
    # Annotate each college name next to its point
    for i, inst in enumerate(df_pca["Institution"]):
        ax.annotate(inst, (df_pca["PC1"][i] + 0.05, df_pca["PC2"][i]), fontsize=9, fontweight='medium')

    ax.set_title("PCA Projection of East Coast Colleges and Universities (Ex 9)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure to outputs/figures/
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fig_ex9_colleges_pca.png")
    plt.savefig(fig_path, dpi=300)
    print(f"[OUTPUT] Figure successfully saved to: {fig_path}")
    plt.show()

if __name__ == "__main__":
    run_exercise_9()