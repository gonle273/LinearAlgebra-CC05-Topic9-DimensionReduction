"""
Master script to execute all experiments and reproduce outputs.
Run command: python run_all.py
"""

from experiments.c3_digits_pca_vs_lda import run_experiment_c3

if __name__ == "__main__":
    print("[PIPELINE] Running all project experiments...")
    run_experiment_c3()
    print("[PIPELINE] Execution completed successfully!")