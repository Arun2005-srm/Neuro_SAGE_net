import csv
import json
from pathlib import Path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def save_predictions(path, rows, classes):
    fields = ["sample_id", "path", "true_class", "pred_class", "confidence", "correct"] + [f"probability_{name}" for name in classes]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row[key] for key in fields[:6]}
            values.update({f"probability_{name}": p for name, p in zip(classes, row["probabilities"])})
            writer.writerow(values)


def save_history_csv(path, history, classes):
    flat = []
    for entry in history:
        row = {"epoch": entry["epoch"], "learning_rates": json.dumps(entry["learning_rates"])}
        for split in ("train", "val"):
            metrics = entry[split]
            row.update({f"{split}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            for name in classes:
                row.update({f"{split}_{name}_{key}": value for key, value in metrics["per_class"][name].items()})
        flat.append(row)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
