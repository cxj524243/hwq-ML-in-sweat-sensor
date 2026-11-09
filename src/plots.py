"""Figure generation (headless, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.model import TrainResult  # noqa: E402


def _class_labels(result: TrainResult) -> list[str]:
    return [str(c) for c in result.encoder.classes_]


def plot_confusion(result: TrainResult, title: str, out: Path) -> None:
    labels = _class_labels(result)
    cm = result.confusion
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(labels) + 3), 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black",
                    fontsize=8,
                )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} - confusion matrix (acc={result.accuracy * 100:.2f}%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_per_class_accuracy(result: TrainResult, title: str, out: Path) -> None:
    labels = _class_labels(result)
    accs = []
    for cls_idx in range(len(labels)):
        mask = result.y_test_encoded == cls_idx
        accs.append(float((result.y_pred[mask] == cls_idx).mean()) if mask.any() else 0.0)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(labels) + 3), 4.2))
    ax.bar(range(len(labels)), accs, color="coral", edgecolor="black", linewidth=0.5)
    ax.axhline(result.accuracy, color="steelblue", ls="--", lw=1.2, label="overall")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall per class")
    ax.set_title(f"{title} - accuracy by concentration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pca(result: TrainResult, title: str, out: Path) -> None:
    n_components = min(result.X_train_scaled.shape[1], 27)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(result.X_train_scaled)
    ev = pca.explained_variance_ratio_

    labels = _class_labels(result)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    for cls_idx, name in enumerate(labels):
        mask = result.y_train_encoded == cls_idx
        if not mask.any():
            continue
        x, y = X_pca[mask, 0], X_pca[mask, 1]
        ax.scatter(
            x,
            y,
            s=26,
            alpha=0.72,
            color=[colors[cls_idx]],
            edgecolor="white",
            linewidth=0.4,
            label=f"{name} (n={int(mask.sum())})",
        )
        if len(x) >= 3:
            vals, vecs = np.linalg.eigh(np.cov(x, y))
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 1.96 * np.sqrt(vals)
            ax.add_patch(
                Ellipse(
                    xy=(float(np.mean(x)), float(np.mean(y))),
                    width=float(width),
                    height=float(height),
                    angle=float(theta),
                    facecolor="none",
                    edgecolor=colors[cls_idx],
                    lw=1.1,
                    ls="--",
                    alpha=0.8,
                )
            )

    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_title(
        f"{title} - PCA of colour-grid features "
        f"(PC1+PC2 = {(ev[0] + ev[1]) * 100:.1f}%)"
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true(result: TrainResult, title: str, unit: str, out: Path) -> None:
    y_true = result.encoder.inverse_transform(result.y_test_encoded).astype(float)
    y_pred = result.y_pred_values.astype(float)
    jitter = 0.012
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.06 or 1.0

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(
        y_true + np.random.normal(0, jitter, len(y_true)),
        y_pred + np.random.normal(0, jitter, len(y_pred)),
        s=34,
        alpha=0.6,
        c="royalblue",
        edgecolors="black",
        linewidths=0.4,
    )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.6)
    ax.set_xlabel(f"Reference ({unit})" if unit else "Reference")
    ax.set_ylabel(f"Predicted ({unit})" if unit else "Predicted")
    ax.set_title(f"{title} - predicted vs reference (MAE={result.mae:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
