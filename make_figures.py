"""Regenerates Figure 5 (confusion matrix) and Figure 6 (accuracy / F1 comparison)."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from run_experiments import CATEGORICAL, NUMERICAL, LABELS, SEED, all_models

ORDER = ["rf", "gb", "ann", "stack", "xgb", "hgb"]
SHORT = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "ann": "ANN",
    "stack": "Stacking Ensemble",
    "xgb": "XGBoost",
    "hgb": "Hist-Gradient\nBoosting",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="simulated_pr_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y = df[CATEGORICAL + NUMERICAL], df["target_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    models = all_models()
    acc, f1, preds = {}, {}, {}
    for key in ORDER:
        models[key].fit(X_train, y_train)
        preds[key] = models[key].predict(X_test)
        acc[key] = accuracy_score(y_test, preds[key])
        f1[key] = f1_score(y_test, preds[key])

    # ---- Figure 5 ----------------------------------------------------------
    cm = confusion_matrix(y_test, preds["stack"])
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.imshow(cm, cmap="Blues")
    names = ["Low Effectiveness (0)", "High Effectiveness (1)"]
    tags = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            shade = "white" if cm[i, j] > cm.max() / 2 else "#1f3b57"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=17, fontweight="bold", color=shade)
            ax.text(j, i + 0.22, tags[i][j], ha="center", va="center",
                    fontsize=10, color=shade)
    ax.set_xticks([0, 1], names, fontsize=9)
    ax.set_yticks([0, 1], names, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title("Confusion Matrix - Stacking Ensemble (Test Set)",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig("figure5_confusion_matrix.png", dpi=200)

    # ---- Figure 6 ----------------------------------------------------------
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    b1 = ax.bar(x - 0.2, [acc[k] for k in ORDER], 0.38, label="Accuracy", color="#1f77b4")
    b2 = ax.bar(x + 0.2, [f1[k] for k in ORDER], 0.38, label="F1-Score", color="#ff7f0e")
    for bars, vals, fmt in ((b1, acc, "{:.1%}"), (b2, f1, "{:.2f}")):
        for bar, key in zip(bars, ORDER):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    fmt.format(vals[key]), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [SHORT[k] for k in ORDER], fontsize=9)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.set_title("Performance Comparison of Machine Learning Models",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("figure6_performance_comparison.png", dpi=200)
    print("wrote figure5_confusion_matrix.png and figure6_performance_comparison.png")


if __name__ == "__main__":
    main()
