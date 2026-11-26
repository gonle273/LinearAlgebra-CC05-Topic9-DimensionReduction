
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ==========================================
# 0. PROJECT PATH
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pca import MyPCA


# ==========================================
# 1. LOAD DATA
# ==========================================

def load_data(filename="data/faces.csv"):

    # Automatically locate CSV from project root
    filepath = Path(filename)

    if not filepath.is_absolute():
        filepath = ROOT / filepath

    if not filepath.exists():
        raise FileNotFoundError(
            f"Cannot find data file: {filepath}"
        )

    # Read photo names
    names = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=1,
        usecols=0,
        dtype=str
    )

    # Read 10 numerical features
    data = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=1,
        usecols=range(1, 11),
        dtype=float
    )

    names = np.atleast_1d(names)
    data = np.atleast_2d(data)

    if data.shape[1] != 10:
        raise ValueError("Expected 10 features per photo.")

    if len(names) != data.shape[0]:
        raise ValueError("Invalid CSV structure.")

    if not np.isfinite(data).all():
        raise ValueError("CSV contains invalid values.")

    print("\nData loaded successfully!")
    print("File:", filepath)
    print("Photos:", len(names))
    print("Features:", data.shape[1])

    return data, names


# ==========================================
# 2. PCA USING GITHUB MyPCA
# ==========================================

def pca_analysis(X):

    pca = MyPCA(n_components=2)

    P = pca.fit_transform(X)

    singular_values = pca.singular_values_
    explained_variance = pca.explained_variance_ratio_

    return P, singular_values, explained_variance


# ==========================================
# 3. RUN PCA FOR EACH CASE
# ==========================================

def run_case(name, X, photo_names):

    P, S, energy = pca_analysis(X)

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Number of photos:", X.shape[0])
    print("Number of features:", X.shape[1])

    print("\nSingular values:")
    print(np.round(S, 4))

    print("\nExplained variance:")
    print(f"PC1: {energy[0] * 100:.2f}%")
    print(f"PC2: {energy[1] * 100:.2f}%")
    print(f"PC1 + PC2: {energy.sum() * 100:.2f}%")

    print("\nPCA Coordinates:")

    for i, (x, y) in enumerate(P):

        print(
            f"{photo_names[i]}: "
            f"PC1 = {x:.4f}, "
            f"PC2 = {y:.4f}"
        )

    return {
        "name": name,
        "P": P,
        "energy": energy,
        "photo_names": photo_names
    }


# ==========================================
# 4. VISUALIZATION
# ==========================================

def plot_results(results, suspects):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6)
    )

    for ax, result in zip(axes, results):

        P = result["P"]
        energy = result["energy"]
        names = result["photo_names"]

        for i, (x, y) in enumerate(P):

            photo_name = names[i]

            color = (
                "red"
                if photo_name in suspects
                else "blue"
            )

            ax.scatter(
                x, y,
                color=color,
                s=90
            )

            ax.annotate(
                photo_name,
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=10
            )

        ax.set_title(
            f"{result['name']}\n"
            f"PC1 = {energy[0]*100:.2f}% | "
            f"PC1 + PC2 = {energy.sum()*100:.2f}%"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

    fig.suptitle(
        "Face Recognition using PCA",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/faces.csv"
    )

    parser.add_argument(
        "--a-cols",
        type=int,
        default=5
    )

    parser.add_argument(
        "--suspects",
        nargs="+",
        default=["F5", "M5"]
    )

    args = parser.parse_args()

    # Load CSV
    data, names = load_data(args.data)

    if not 2 <= args.a_cols < len(names):
        raise ValueError("Invalid number of photos in Case A.")

    # Case A: First 5 photos
    A = data[:args.a_cols, :]
    names_A = names[:args.a_cols]

    result_A = run_case(
        f"Case A (Photos 1-{len(names_A)})",
        A,
        names_A
    )

    # Case B: All photos
    result_B = run_case(
        f"Case B (All {len(names)} photos)",
        data,
        names
    )

    # Display PCA results
    plot_results(
        [result_A, result_B],
        args.suspects
    )


if __name__ == "__main__":
    main()
