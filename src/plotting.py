"""
Plotting utilities for dimensionality reduction visualizations.
"""

import matplotlib.pyplot as plt


def plot_side_by_side_2d(
    X_pca, X_lda, y, class_labels,
    title_pca="PCA 2D Projection", 
    title_lda="LDA 2D Projection",
    save_path=None
):
    """
    Generate side-by-side scatter plots for PCA vs. LDA 2D projections.
    
    Parameters:
    -----------
    X_pca : np.ndarray of shape (N, 2)
    X_lda : np.ndarray of shape (N, 2)
    y : np.ndarray of shape (N,)
    class_labels : dict, mapping class index to display name (e.g., {0: 'Digit 0'})
    save_path : str or None, base path without extension to save both .png and .pdf
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    unique_classes = list(class_labels.keys())

    # --- Left: PCA Plot ---
    for i, c in enumerate(unique_classes):
        mask = (y == c)
        color = colors[i % len(colors)]
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, label=class_labels[c],
            alpha=0.8, edgecolors='k', s=45
        )
    axes[0].set_title(title_pca, fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # --- Right: LDA Plot ---
    for i, c in enumerate(unique_classes):
        mask = (y == c)
        color = colors[i % len(colors)]
        axes[1].scatter(
            X_lda[mask, 0], X_lda[mask, 1],
            c=color, label=class_labels[c],
            alpha=0.8, edgecolors='k', s=45
        )
    axes[1].set_title(title_lda, fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Linear Discriminant 1")
    axes[1].set_ylabel("Linear Discriminant 2")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.savefig(f"{save_path}.pdf")
        print(f"[PLOT] Figures saved to: {save_path}.png and {save_path}.pdf")

    plt.show()