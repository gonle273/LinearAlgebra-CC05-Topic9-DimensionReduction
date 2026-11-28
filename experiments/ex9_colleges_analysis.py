"""Exercise 9 (Section 3.9): Standardization -> PCA to R^2 -> Projections -> Two-Class LDA.

Uses shared modules `src.pca.MyPCA` and `src.lda.MyLDA` and saves figures to `outputs/figures/`.
Compatible with Google Colab (cloned GitHub repo), Jupyter notebooks, and CLI execution.
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- RESOLVE PROJECT ROOT ----------------------
start_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
search_candidates = [
    start_dir,
    start_dir.parent,
    Path("/content/repo"),
    Path("/content"),
]

project_root = start_dir
for base in search_candidates:
    if not base.exists():
        continue
    if (base / "src").is_dir():
        project_root = base
        break
    for match in list(base.glob("*/src")) + list(base.glob("*/*/src")):
        if match.is_dir():
            project_root = match.parent
            break
    if (project_root / "src").is_dir():
        break

if not (project_root / "src").is_dir():
    raise FileNotFoundError(
        "Could not find 'src/' folder. Clone the GitHub repository first (e.g., "
        "!git clone <repo_url> /content/repo) and run again."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pca import MyPCA
from src.lda import MyLDA

np.set_printoptions(precision=3, suppress=True)

# ---------------------- DATA DEFINITION ----------------------
names = [
    "Bucknell",
    "Davidson",
    "Florida State",
    "Georgia Tech",
    "Lafayette",
    "UNC",
    "Richmond",
    "Virginia",
    "Villanova",
    "William & Mary",
]

# Columns: out-of-state tuition, undergraduates, acceptance rate, graduation rate
X = np.array(
    [
        [56092, 4900, 0.31, 0.90],
        [51447, 1843, 0.20, 0.93],
        [18786, 32812, 0.50, 0.80],
        [33020, 16049, 0.23, 0.85],
        [53630, 2642, 0.31, 0.90],
        [35170, 19117, 0.24, 0.90],
        [52610, 3227, 0.33, 0.88],
        [49032, 11786, 0.27, 0.93],
        [56730, 10983, 0.29, 0.91],
        [45272, 6377, 0.34, 0.90],
    ],
    dtype=float,
)
uva_index = names.index("Virginia")

# ---------------------- PART (a): STANDARDIZATION & PCA ----------------------
# Standardize features using sample standard deviation (ddof=1)
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

# Fit shared MyPCA model (Z already has zero mean, so internal centering leaves Z unchanged)
pca = MyPCA(n_components=2)
T = pca.fit_transform(Z)

# Enforce deterministic sign convention: largest absolute loading positive
if hasattr(pca, "components_"):
    V = pca.components_.T
    for j in range(T.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1.0
            T[:, j] *= -1.0

print("=" * 60)
print(f"Using repository root: {project_root}")
if hasattr(pca, "singular_values_"):
    print("SINGULAR VALUES:", pca.singular_values_)
if hasattr(pca, "explained_variance_ratio_"):
    print("PROPORTION OF VARIANCE EXPLAINED:", pca.explained_variance_ratio_)
    print(
        "CUMULATIVE VARIANCE (PC1 + PC2):",
        round(float(np.sum(pca.explained_variance_ratio_[:2])), 3),
    )

# Euclidean distance to Virginia in the 2D PCA plane
dist_to_uva = np.linalg.norm(T - T[uva_index], axis=1)
print("\n(a) Distance to Virginia in R^2:")
print({name: round(float(dist_to_uva[i]), 3) for i, name in enumerate(names) if i != uva_index})

# ---------------------- PART (b): PC1 COORDINATE ----------------------
print("\n(b) Absolute difference along PC1 axis:")
pc1_diff = np.abs(T[:, 0] - T[uva_index, 0])
print({name: round(float(pc1_diff[i]), 3) for i, name in enumerate(names) if i != uva_index})

# ---------------------- PART (c): PROJECTION ONTO LINE ----------------------
# Line: y = x/4 - 1/2 -> direction vector u = (4, 1), point p0 = (0, -0.5)
u = np.array([4.0, 1.0])
p0 = np.array([0.0, -0.5])
u_unit = u / np.linalg.norm(u)

# Signed coordinate along the unit direction of the line
t = (T - p0) @ u_unit

print("\n(c) Absolute difference in position along the line:")
print({name: round(float(np.abs(t[i] - t[uva_index])), 3) for i, name in enumerate(names) if i != uva_index})

# ---------------------- PART (d): GROUPING & SCATTER MATRICES ----------------------
# Group 1: < 10,000 undergraduates (label 0) | Group 2: >= 10,000 undergraduates (label 1)
group1_mask = X[:, 1] < 10000
y = np.where(group1_mask, 0, 1)

T_g1 = T[group1_mask]
T_g2 = T[~group1_mask]

mean_g1 = T_g1.mean(axis=0)
mean_g2 = T_g2.mean(axis=0)

S_W = (T_g1 - mean_g1).T @ (T_g1 - mean_g1) + (T_g2 - mean_g2).T @ (T_g2 - mean_g2)
S_B = np.outer(mean_g1 - mean_g2, mean_g1 - mean_g2)

print("\n" + "=" * 60)
print("(d) CLASS MEANS IN R^2:")
print("  Mean (Group 1, <10k):", mean_g1)
print("  Mean (Group 2, >=10k):", mean_g2)
print("\nWithin-class scatter matrix S_W:\n", S_W)
print("\nBetween-class scatter matrix S_B:\n", S_B)

# ---------------------- PART (e): FISHER LDA VIA MyLDA ----------------------
lda = MyLDA(n_components=1)
_ = lda.fit_transform(T, y).ravel()

# Extract Fisher discriminant direction from MyLDA
if hasattr(lda, "scalings_"):
    v = lda.scalings_[:, 0]
elif hasattr(lda, "components_"):
    v = lda.components_[0]
elif hasattr(lda, "w_"):
    v = lda.w_.ravel()
else:
    v = np.linalg.solve(S_W, mean_g1 - mean_g2)

# Orient v consistently from Group 2 (>=10k) toward Group 1 (<10k)
if np.dot(v, mean_g1 - mean_g2) < 0:
    v = -v
v_unit = v / np.linalg.norm(v)


def rayleigh_fisher(w: np.ndarray) -> float:
    """Compute Fisher ratio (w^T S_B w) / (w^T S_W w) for nonzero vector w."""
    return float((w.T @ S_B @ w) / (w.T @ S_W @ w))


print("\n" + "=" * 60)
print("(e) LDA DIRECTION IN R^2:")
print("  v_unit           =", v_unit)
print("  J(LDA)           =", round(rayleigh_fisher(v_unit), 3))
print("  J(PC1 axis)      =", round(rayleigh_fisher(np.array([1.0, 0.0])), 3))
print("  J(PC2 axis)      =", round(rayleigh_fisher(np.array([0.0, 1.0])), 3))
print("  J(line dir)      =", round(rayleigh_fisher(u_unit), 3))

angle_deg = float(np.degrees(np.arccos(np.clip(np.abs(v_unit[0]), 0.0, 1.0))))
print("  Angle between LDA and PC1 (deg):", round(angle_deg, 3))

# Nearest-mean classification on each 1D projection
print("\nMisclassification by projection (nearest-mean rule):")
for label, scores in [("PC1", T[:, 0]), ("Line", t), ("LDA", T @ v_unit)]:
    mean_s1 = scores[group1_mask].mean()
    mean_s2 = scores[~group1_mask].mean()
    threshold = 0.5 * (mean_s1 + mean_s2)
    predictions = scores > threshold if mean_s1 > threshold else scores < threshold
    misclassified = [names[i] for i in range(len(names)) if predictions[i] != group1_mask[i]]
    print(f"  {label:4s}: Misclassified -> {misclassified if misclassified else 'None'}")

# ---------------------- ROBUSTNESS CHECK: LDA IN FULL 4-D SPACE ----------------------
lda_4d = MyLDA(n_components=1)
scores_4d = lda_4d.fit_transform(Z, y).ravel()

print("\n" + "=" * 60)
print("LDA IN FULL 4-DIMENSIONAL STANDARDIZED SPACE (via MyLDA):")
print(
    "  Group 1 (<10k) score range : [{:.3f}, {:.3f}]".format(
        scores_4d[group1_mask].min(), scores_4d[group1_mask].max()
    )
)
print(
    "  Group 2 (>=10k) score range: [{:.3f}, {:.3f}]".format(
        scores_4d[~group1_mask].min(), scores_4d[~group1_mask].max()
    )
)

# ---------------------- VISUALIZATION: PCA PLANE WITH LDA ----------------------
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    T_g1[:, 0],
    T_g1[:, 1],
    c="steelblue",
    label="Group 1: < 10,000 Undergraduates",
    s=90,
    edgecolors="black",
    alpha=0.85,
)
ax.scatter(
    T_g2[:, 0],
    T_g2[:, 1],
    c="crimson",
    label="Group 2: >= 10,000 Undergraduates",
    s=90,
    edgecolors="black",
    alpha=0.85,
)

offsets = {
    "Bucknell": (0.04, 0.03),
    "Davidson": (0.04, 0.03),
    "Florida State": (0.04, -0.06),
    "Georgia Tech": (0.04, 0.03),
    "Lafayette": (0.04, -0.05),
    "UNC": (0.04, 0.03),
    "Richmond": (0.04, -0.05),
    "Virginia": (0.04, 0.03),
    "Villanova": (0.04, -0.05),
    "William & Mary": (0.04, 0.03),
}
for idx, school in enumerate(names):
    dx, dy = offsets.get(school, (0.04, 0.02))
    ax.annotate(school, (T[idx, 0] + dx, T[idx, 1] + dy), fontsize=9)

ax.scatter(
    mean_g1[0],
    mean_g1[1],
    c="darkblue",
    marker="X",
    s=190,
    linewidths=1.8,
    edgecolors="white",
    zorder=6,
    label="Centroid (<10k)",
)
ax.scatter(
    mean_g2[0],
    mean_g2[1],
    c="darkred",
    marker="X",
    s=190,
    linewidths=1.8,
    edgecolors="white",
    zorder=6,
    label="Centroid (>=10k)",
)

arrow_scale = 1.6
ax.quiver(
    0.0,
    0.0,
    v_unit[0] * arrow_scale,
    v_unit[1] * arrow_scale,
    angles="xy",
    scale_units="xy",
    scale=1.0,
    color="forestgreen",
    width=0.006,
    zorder=5,
    label="LDA Direction",
)
ax.quiver(
    0.0,
    0.0,
    1.2,
    0.0,
    angles="xy",
    scale_units="xy",
    scale=1.0,
    color="dimgray",
    width=0.004,
    alpha=0.75,
    zorder=4,
    label="PC1 Axis",
)

ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
ax.set_xlabel("PC1", fontsize=12, fontweight="bold")
ax.set_ylabel("PC2", fontsize=12, fontweight="bold")
ax.set_title(
    "PCA Projection Plane with LDA Direction and Group Means",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.legend(loc="best", fontsize=9.5)
ax.grid(True, alpha=0.35, linestyle="--")
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()

output_path = project_root / "outputs/figures/ex9_pca_lda.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nSaved figure to: {output_path}")

plt.show()