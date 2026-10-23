import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]


def main():
    benchmark = json.loads((ROOT / "reports/benchmark.json").read_text(encoding="utf-8"))
    keys = ["precision", "recall", "f1", "roc_auc"]
    baseline = [benchmark["baseline"][key] for key in keys]
    forest = [benchmark["random_forest"][key] for key in keys]
    matrix = np.array(benchmark["random_forest"]["confusion_matrix"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#0f172a",
        "xtick.color": "#334155", "ytick.color": "#334155",
    })
    fig, (scores, confusion) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                         gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Churn model · held-out test results", x=0.06, y=0.96,
                 ha="left", fontsize=19, weight="bold")
    positions = np.arange(len(keys))
    width = 0.34
    for values, offset, label, color in [
        (baseline, -width / 2, "Dummy baseline (prior)", "#94a3b8"),
        (forest, width / 2, "Tuned Random Forest", "#0f766e"),
    ]:
        bars = scores.bar(positions + offset, values, width, color=color, label=label, zorder=3)
        scores.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5, fontsize=10)
    scores.set_xticks(positions, ["Precision", "Recall", "F1", "ROC-AUC"])
    scores.set_ylim(0, 1.12)
    scores.set_yticks(np.arange(0, 1.01, 0.2))
    scores.set_ylabel("Score")
    scores.grid(axis="y", color="#e2e8f0", zorder=0)
    scores.spines["left"].set_color("#cbd5e1")
    scores.spines["bottom"].set_color("#cbd5e1")
    scores.legend(loc="upper left", bbox_to_anchor=(0, -0.15), frameon=False, fontsize=10)

    palette = LinearSegmentedColormap.from_list("teal", ["#f0fdfa", "#0f766e"])
    confusion.imshow(matrix, cmap=palette, vmin=0, vmax=matrix.max())
    confusion.set_title("Random Forest confusion matrix", pad=13, fontsize=12, weight="bold")
    confusion.set_xticks([0, 1], ["Retained", "Churned"])
    confusion.set_yticks([0, 1], ["Retained", "Churned"])
    confusion.set_xlabel("Predicted class", labelpad=9)
    confusion.set_ylabel("Actual class", labelpad=9)
    for row in range(2):
        for col in range(2):
            confusion.text(col, row, str(matrix[row, col]), ha="center", va="center",
                           fontsize=24, weight="bold",
                           color="white" if matrix[row, col] > matrix.max() / 2 else "#134e4a")
    for spine in confusion.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.065, right=0.96, top=0.79, bottom=0.25, wspace=0.4)
    fig.text(0.06, 0.055,
             f"Source: reports/benchmark.json  |  UCI Iranian Churn  |  "
             f"{benchmark['test_rows']} test records  |  Seed {benchmark['seed']}",
             fontsize=10, color="#475569")
    destination = ROOT / "docs/assets"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "churn-benchmark.png", dpi=180, facecolor="white",
                metadata={"Title": "Held-out churn model benchmark"})
    plt.close(fig)
    print("Saved docs/assets/churn-benchmark.png from reports/benchmark.json")


if __name__ == "__main__":
    main()
