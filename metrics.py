"""Dataset-level metrics; class accuracy means recall within that true class."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def classification_metrics(labels, probabilities, classes, loss=None):
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if labels.size == 0 or probabilities.shape != (labels.size, len(classes)):
        raise ValueError("Metrics require nonempty labels and [samples, classes] probabilities")
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=np.arange(len(classes)), zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=np.arange(len(classes)))
    per_class = {}
    for idx, name in enumerate(classes):
        tp = int(cm[idx, idx])
        fp, fn = int(cm[:, idx].sum() - tp), int(cm[idx].sum() - tp)
        tn = len(labels) - tp - fp - fn
        binary_labels = (labels == idx).astype(int)
        auc = float(roc_auc_score(binary_labels, probabilities[:, idx])) if len(np.unique(binary_labels)) == 2 else None
        per_class[name] = {"accuracy": float(recall[idx]), "precision": float(precision[idx]), "recall": float(recall[idx]),
                           "f1": float(f1[idx]), "support": int(support[idx]), "specificity": tn / (tn + fp) if tn + fp else 0.0,
                           "one_vs_rest_accuracy": (tp + tn) / len(labels), "roc_auc": auc}
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    one_hot = np.eye(len(classes))[labels]
    auc_values = [m["roc_auc"] for m in per_class.values() if m["roc_auc"] is not None]
    return {"loss": float(loss) if loss is not None else None, "accuracy": float(correct.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.average(f1, weights=support)), "macro_roc_auc": float(np.mean(auc_values)) if auc_values else None,
            "ece": float(ece), "brier_score": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
            "samples": len(labels), "per_class": per_class, "confusion_matrix": cm.tolist()}


def format_epoch(epoch, total, train, val):
    def summary(metrics):
        return (f"Acc {metrics['accuracy']:.2%} / Loss {metrics['loss']:.4f}")
    lines = [f"ep[{epoch}/{total}] Train: {summary(train)} || Val: {summary(val)}"]
    lines[0] += f" | Precision {val['macro_precision']:.2%} | F1 {val['macro_f1']:.2%}"
    lines.append("Val class acc: " + " | ".join(
        f"{name} {metrics['accuracy']:.2%}" for name, metrics in val["per_class"].items()
    ))
    return "\n".join(lines)
