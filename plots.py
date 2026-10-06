"""Save learning curves, class metrics, confusion matrices, ROC/PR, and calibration."""
from pathlib import Path
import os
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc


def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _names(classes):
    return [textwrap.fill(str(name), 18) for name in classes]


def training_curves(history, directory, classes, dpi=160):
    directory = Path(directory)
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, key, title in zip(axes.flat, ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"),
                                ("Loss", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "Balanced accuracy")):
        for split, label in (("train", "Train"), ("val", "Fold validation")):
            axis.plot(epochs, [entry[split][key] for entry in history], label=label)
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=.25)
        axis.legend()
    _save(fig, directory / "training_curves.png", dpi)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, split, title in zip(axes, ("train", "val"), ("Train class accuracy (recall)", "Fold validation class accuracy (recall)")):
        for name in classes:
            axis.plot(epochs, [entry[split]["per_class"][name]["accuracy"] for entry in history], label=name)
        axis.set(title=title, xlabel="Epoch", ylim=(0, 1.02))
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    _save(fig, directory / "class_accuracy_curves.png", dpi)


def evaluation_plots(metrics, rows, classes, directory, dpi=160):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["confusion_matrix"])
    count = len(classes)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, count * 1.7), max(5, count * .8)))
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    for axis, values, title in zip(axes, (cm, normalized), ("Confusion matrix", "Confusion matrix (row normalized)")):
        axis.imshow(values, cmap="Blues")
        axis.set(xticks=range(count), yticks=range(count), xticklabels=_names(classes), yticklabels=_names(classes),
                 xlabel="Predicted class", ylabel="Actual class", title=title)
        axis.tick_params(axis="x", rotation=45)
        if count <= 20:
            for i in range(count):
                for j in range(count):
                    text = f"{values[i, j]:.2f}" if values.dtype.kind == "f" else str(values[i, j])
                    axis.text(j, i, text, ha="center", va="center", fontsize=8,
                              color="white" if values[i, j] > values.max() / 2 else "black")
    _save(fig, directory / "confusion_matrix.png", dpi)
    fig, axis = plt.subplots(figsize=(max(9, count * 1.1), 5))
    x = np.arange(count)
    for offset, key in zip((-.24, 0, .24), ("precision", "recall", "f1")):
        axis.bar(x + offset, [metrics["per_class"][name][key] for name in classes], width=.24, label=key.title())
    axis.set(xticks=x, xticklabels=_names(classes), ylim=(0, 1.05), title="Class metrics (recall = class accuracy)")
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    _save(fig, directory / "class_metrics.png", dpi)
    labels = np.array([row["true_index"] for row in rows])
    probs = np.array([row["probabilities"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, name in enumerate(classes):
        binary = (labels == idx).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, idx])
        axes[0].plot(fpr, tpr, label=f"{name}: {auc(fpr, tpr):.3f}")
        precision, recall, _ = precision_recall_curve(binary, probs[:, idx])
        axes[1].plot(recall, precision, label=f"{name}: {average_precision_score(binary, probs[:, idx]):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0].set(title="One-vs-rest ROC (AUC)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="One-vs-rest precision-recall (AP)", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    _save(fig, directory / "roc_pr_curves.png", dpi)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    centers, accuracies = [], []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            centers.append(confidence[mask].mean())
            accuracies.append(correct[mask].mean())
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.plot(centers, accuracies, "o-", label=f"ECE={metrics['ece']:.4f}")
    axis.set(xlabel="Mean confidence", ylabel="Observed accuracy", title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    _save(fig, directory / "calibration.png", dpi)


def fold_comparison(results, directory, dpi=160):
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    axis.bar(x - .18, [r["cv_metrics"]["macro_f1"] for r in results], .36, label="Fold validation")
    if all(r.get("holdout_metrics") is not None for r in results):
        axis.bar(x + .18, [r["holdout_metrics"]["macro_f1"] for r in results], .36, label="15% selection validation")
    axis.set(xticks=x, xticklabels=[f"Fold {r['fold']}" for r in results], ylabel="Macro F1", ylim=(0, 1.05), title="Fold checkpoint comparison")
    axis.legend()
    _save(fig, Path(directory) / "fold_comparison.png", dpi)
