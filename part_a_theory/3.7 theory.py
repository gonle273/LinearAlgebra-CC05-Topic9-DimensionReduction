import numpy as np
import matplotlib.pyplot as plt
from src.pca import MyPCA

tribes = [
    "Ojibawa1", "Ojibawa2", "Ojibawa3", "Ojibawa4", "Ojibawa5",
    "MicMac1", "MicMac2", "MicMac3", "Passamaquody1", "Passamaquody2"
]

# Exact 10x16 binary matrix from Section 3.7 (page 106)
M = np.array([
    [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1]
], dtype=float)

# Fit and project into R^2 using MyPCA from src/pca.py
pca = MyPCA(n_components=2)
coords = pca.fit_transform(M)

s1, s2 = pca.singular_values_
var1, var2 = pca.explained_variance_ratio_ * 100
print(f"sigma_1 = {s1:.5f}, sigma_2 = {s2:.5f}")
print(f"Explained Variance: PC1 = {var1:.2f}%, PC2 = {var2:.2f}% -> Total = {var1 + var2:.2f}%")

plt.figure(figsize=(8, 5))
plt.scatter(coords[:5, 0], coords[:5, 1], c='black', marker='o', s=75, label='Ojibawa')
plt.scatter(coords[5:8, 0], coords[5:8, 1], c='black', marker='^', s=80, label='MicMac')
plt.scatter(coords[8:, 0], coords[8:, 1], c='black', marker='s', s=75, label='Passamaquody')
for i, txt in enumerate(tribes):
    plt.annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=8,
                 xytext=(4, 4), textcoords='offset points')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title(f"Woodland Tribes PCA in R^2 (Variance Retained: {var1 + var2:.2f}%)")
plt.xlabel(f"PC1 ({var1:.2f}% variance)")
plt.ylabel(f"PC2 ({var2:.2f}% variance)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
