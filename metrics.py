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
        return (f"Acc {metrics['accuracy']:.4f} / Loss {metrics['loss']:.4f} | "
                f"Precision {metrics['macro_precision']:.4f} | Recall {metrics['macro_recall']:.4f} | "
                f"F1 {metrics['macro_f1']:.4f} | BalancedAcc {metrics['balanced_accuracy']:.4f}")
    lines = [f"ep[{epoch}/{total}] Train: {summary(train)} || Val: {summary(val)}"]
    for name in train["per_class"]:
        t, v = train["per_class"][name], val["per_class"][name]
        lines.append(f"  {name}: Train class acc {t['accuracy']:.4f}, P {t['precision']:.4f}, R {t['recall']:.4f}, F1 {t['f1']:.4f}"
                     f" | Val class acc {v['accuracy']:.4f}, P {v['precision']:.4f}, R {v['recall']:.4f}, F1 {v['f1']:.4f} (n={v['support']})")
    return "\n".join(lines)
