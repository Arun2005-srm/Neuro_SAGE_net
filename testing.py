"""Evaluate a selected checkpoint against the saved, untouched 15% test partition."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from data.contracts import SampleRecord
from data.loaders import build_loader
from feature_cache import prepare_cache
from losses import ClassificationLoss
from models import NeuroSAGENet
from plots import evaluation_plots
from reporting import save_predictions, write_json
from runtime import setup_runtime
from trainer import run_epoch
from visualization import save_prediction_images


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = NeuroSAGENet(checkpoint["config"], len(checkpoint["classes"]), load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def evaluate_run(run_dir, repeat=False):
    run_dir = Path(run_dir).resolve()
    if (run_dir / "test_metrics.json").exists() and not repeat:
        raise ValueError("Test was already evaluated. Use --repeat only for an explicit reproducibility check, not model tuning.")
    split_data = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best_model.pt"
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg, classes = metadata["config"], metadata["classes"]
    if classes != split_data["classes"]:
        raise ValueError("Checkpoint and split class mapping disagree")
    records = [SampleRecord(**row) for row in split_data["partitions"]["test"]]
    for record in records:
        if hashlib.sha256(Path(record.path).read_bytes()).hexdigest()[:24] != record.sample_id:
            raise ValueError(f"Test image changed since splitting: {record.path}")
    device = setup_runtime(cfg["runtime"])
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    cache_dir = prepare_cache(model, records, cfg, device)
    loader = build_loader(records, cfg, cache_dir=cache_dir)
    metrics, rows = run_epoch(model, loader, ClassificationLoss(cfg["loss"]), device, classes,
                             amp_enabled=cfg["runtime"]["use_amp"] and device.type == "cuda")
    metrics.update({"selected_fold": checkpoint["fold"], "selected_epoch": checkpoint["epoch"],
                    "timing_scope": "cached-feature evaluation including loading" if cache_dir else "image-to-prediction evaluation including loading",
                    "repeat_evaluation": repeat})
    write_json(run_dir / "test_metrics.json", metrics)
    write_json(run_dir / "test_predictions.json", rows)
    save_predictions(run_dir / "test_predictions.csv", rows, classes)
    evaluation_plots(metrics, rows, classes, run_dir / "test_plots", cfg["evaluation"]["dpi"])
    save_prediction_images(rows, metrics, cfg["evaluation"], run_dir / "test_visualizations", model.backbone.grid_size)
    print(f"Test | Acc {metrics['accuracy']:.4f} | Precision {metrics['macro_precision']:.4f} | Recall {metrics['macro_recall']:.4f} | Macro F1 {metrics['macro_f1']:.4f}", flush=True)
    for name, values in metrics["per_class"].items():
        print(f"  {name}: class acc {values['accuracy']:.4f} | P {values['precision']:.4f} | R {values['recall']:.4f} | F1 {values['f1']:.4f} | n={values['support']}", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.run_dir, args.repeat)
