import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from src.pca import MyPCA



def run_pca_and_report(M, labels, title, save_filename, groups=None, group_title="Region"):
    m, n = M.shape
    print(f"1. Feature matrix M shape: {m} stories x {n} attributes (min={M.min()}, max={M.max()})")

    # PCA via Truncated SVD using MyPCA from src/pca.py
    pca = MyPCA(n_components=2)
    coords = pca.fit_transform(M)

    s1, s2 = pca.singular_values_
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"2. Dominant singular values (centered): sigma_1 = {s1:.5f}, sigma_2 = {s2:.5f}")
    print(f"   Variance retained in R^2: PC1 = {var1:.2f}%, PC2 = {var2:.2f}% -> Total = {var1 + var2:.2f}%")

    # Pairwise Euclidean distances in R^2
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            l1 = f"{labels[i]} ({groups[i]})" if groups is not None else labels[i]
            l2 = f"{labels[j]} ({groups[j]})" if groups is not None else labels[j]
            pairs.append((dist_matrix[i, j], l1, l2))
    pairs.sort(key=lambda x: x[0])

    print("\n3. TOP 10 CLOSEST STORY PAIRS IN R^2:")
    for rank, (d, s1_name, s2_name) in enumerate(pairs[:10], 1):
        print(f"   {rank:2d}. {s1_name} <--> {s2_name} | d = {d:.5f}")

    # Geographic validation (for Exercise 7)
    if groups is not None:
        intra, inter = [], []
        for i in range(m):
            for j in range(i + 1, m):
                if groups[i] == groups[j]:
                    intra.append(dist_matrix[i, j])
                else:
                    inter.append(dist_matrix[i, j])
        print(f"\n4. GEOGRAPHIC VALIDATION ({group_title}):")
        print(f"   - Mean intra-region distance (d_intra): {np.mean(intra):.5f}")
        print(f"   - Mean inter-region distance (d_inter): {np.mean(inter):.5f}")

    # Plot and save 300 DPI figure
    plt.figure(figsize=(12, 8))
    if groups is not None:
        unique_groups = pd.unique(groups)
        cmap = plt.get_cmap('tab10')
        for idx, grp in enumerate(unique_groups):
            mask = (groups == grp)
            plt.scatter(coords[mask, 0], coords[mask, 1], label=grp,
                        color=cmap(idx % 10), s=65, alpha=0.85, edgecolors='black', linewidth=0.5)
        plt.legend(title=group_title, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], c='#2b5c8f', s=70, alpha=0.85, edgecolors='black', linewidth=0.5)

    for i, name in enumerate(labels):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=7.5, alpha=0.85,
                     xytext=(4, 2), textcoords='offset points')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.title(f"{title}\nPCA in R^2 (Variance Retained: {var1 + var2:.1f}%)", fontsize=12, fontweight='bold')
    plt.xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=10)
    plt.ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()


# --- EXERCISE 7: 86 STAR HUSBAND TALES ---
path_ex7 = os.path.join(PROJECT_ROOT, "data", "starhusbandbook.csv")
df7 = pd.read_csv(path_ex7).dropna(how='all').reset_index(drop=True)
text_cols7 = df7.select_dtypes(exclude=[np.number]).columns.tolist()
story_col7 = max(text_cols7, key=lambda c: df7[c].nunique())
region_col7 = min(text_cols7, key=lambda c: abs(df7[c].nunique() - 9))

raw_names7 = df7[story_col7].astype(str).str.strip()
counts = raw_names7.groupby(raw_names7).cumcount() + 1
totals = raw_names7.groupby(raw_names7).transform('count')
labels7 = np.where(totals > 1, raw_names7 + "_" + counts.astype(str), raw_names7)
groups7 = df7[region_col7].astype(str).str.strip().values

num_df7 = df7.select_dtypes(include=[np.number]).fillna(0)
drop7 = [c for c in num_df7.columns if num_df7[c].max() > 1 or any(k in c.lower() for k in ['total', 'sum', 'id', 'unnamed'])]
num_df7 = num_df7.drop(columns=drop7, errors='ignore')

run_pca_and_report(
    M=num_df7.values.astype(float), labels=labels7,
    title="Exercise 7: PCA Clustering of 86 Star Husband Tales",
    save_filename=os.path.join(PROJECT_ROOT, "outputs", "figures", "exercise7_star_husband_mypca.png"),
    groups=groups7, group_title=region_col7
)

# --- EXERCISE 8: 31 CINDERELLA STORIES (CLEANED & TRANSPOSED) ---
path_ex8 = os.path.join(PROJECT_ROOT, "data", "cinderellastories.csv")
df8 = pd.read_csv(path_ex8).dropna(how='all').reset_index(drop=True)
num_df8 = df8.select_dtypes(include=[np.number]).fillna(0)
drop_cols8 = [c for c in num_df8.columns if any(k in str(c).lower() for k in ['total', 'sum', 'unnamed'])]
num_df8 = num_df8.drop(columns=drop_cols8, errors='ignore')

first_col_str = df8.iloc[:, 0].astype(str).str.lower()
valid_motif_rows = (num_df8.max(axis=1) <= 1) & (~first_col_str.str.contains('total|sum|count|nan', regex=True))
num_df8_clean = num_df8.loc[valid_motif_rows]

M8_stories = num_df8_clean.T.values.astype(float)
story_names8 = [str(c).strip() for c in num_df8_clean.columns]

run_pca_and_report(
    M=M8_stories, labels=story_names8,
    title="Exercise 8: PCA Clustering of 31 Cinderella Stories",
    save_filename=os.path.join(PROJECT_ROOT, "outputs", "figures", "exercise8_cinderella_mypca.png"),
    groups=None
)
