"""Run five folds for twenty epochs each and evaluate the selected model once."""
import argparse
import gc
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from config import load_config
from data.discovery import discover_records
from data.loaders import build_loader
from data.splitting import make_folds, save_splits, split_records
from feature_cache import prepare_cache
from losses import ClassificationLoss
from metrics import classification_metrics, format_epoch
from models import NeuroSAGENet
from plots import evaluation_plots, fold_comparison, training_curves
from reporting import save_history_csv, save_predictions, write_json
from runtime import seed_everything, setup_runtime
from trainer import run_epoch
from visualization import save_prediction_images


def _score(metrics, key):
    return -metrics["loss"] if key == "loss" else metrics[key]


def main(config_path, dry_run=False, skip_test=False):
    cfg = load_config(config_path)
    records, classes = discover_records(cfg["dataset"])
    partitions = split_records(records, cfg["dataset"]["split"], classes)
    cv_pool = partitions["train"] if cfg["training"]["cv_pool"] == "train" else partitions["train"] + partitions["val"]
    fold_cfg = {**cfg["training"], "seed": cfg["dataset"]["split"]["seed"]}
    folds = make_folds(cv_pool, fold_cfg, classes, cfg["dataset"]["split"]["strategy"] == "group")
    sizes = {name: len(part) for name, part in partitions.items()}
    print(f"Classes: {classes}\n70/15/15 partition counts: {sizes}\nCV pool: {cfg['training']['cv_pool']} ({len(cv_pool)} images)", flush=True)
    if dry_run:
        print("Dataset and five folds validated. No training or output files created.", flush=True)
        return {"partitions": sizes, "classes": classes}
    output = Path(cfg["training"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new training.output_dir.")
    output.mkdir(parents=True, exist_ok=True)
    def log(message):
        print(message, flush=True)
        with (output / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    save_splits(output / "splits.json", partitions, folds, classes, cfg["training"]["cv_pool"])
    write_json(output / "class_mapping.json", {name: i for i, name in enumerate(classes)})
    write_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
                                             "cuda_available": torch.cuda.is_available(), "config_source": str(Path(config_path).resolve())})
    device = setup_runtime(cfg["runtime"])
    amp = cfg["runtime"]["use_amp"] and device.type == "cuda"
    train_cfg = cfg["training"]
    log(f"Device: {device} | AMP: {amp} | Exactly 5 folds x 20 epochs; checkpoint selection uses {train_cfg['selection_metric']}")
    log(f"Partition counts: {sizes} | Classes: {classes}")
    if cfg["dataset"]["split"]["strategy"] == "group":
        log("Group integrity is enforced; 70/15/15 image counts are approximate when group sizes differ.")
    if train_cfg["cv_pool"] == "train":
        log("Each fold validates after every epoch. Independent 15% validation selects the final fold checkpoint; test remains untouched.")
    else:
        log("CV spans the 85% development pool. The original 15% validation is part of CV, not an independent selection holdout.")
    results, oof_rows = [], []
    cache_dir = None
    for fold_number, (train_records, val_records) in enumerate(folds, 1):
        fold_dir = output / f"fold_{fold_number}"
        fold_dir.mkdir()
        # Identical frozen prefix across folds is required for valid shared caching.
        seed_everything(cfg["runtime"]["seed"], cfg["runtime"]["deterministic"])
        model = NeuroSAGENet(cfg, len(classes)).to(device)
        if fold_number == 1:
            # No test features are extracted during training/model selection.
            cache_dir = prepare_cache(model, partitions["train"] + partitions["val"], cfg, device)
            log(f"Parameters: total={sum(p.numel() for p in model.parameters()):,}; trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        seed_everything(cfg["runtime"]["seed"] + fold_number, cfg["runtime"]["deterministic"])
        train_loader = build_loader(train_records, cfg, True, cfg["runtime"]["seed"] + fold_number, cache_dir)
        val_loader = build_loader(val_records, cfg, cache_dir=cache_dir)
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        other_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("backbone.")]
        optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": train_cfg["backbone_learning_rate"]},
                                       {"params": other_params, "lr": train_cfg["learning_rate"]}], weight_decay=train_cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_cfg["epochs"]) if train_cfg["scheduler"] == "cosine" else None
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        criterion = ClassificationLoss(cfg["loss"])
        history, best_score = [], -float("inf")
        log(f"\nFold [{fold_number}/5] | Train images {len(train_records)} | Fold validation images {len(val_records)}")
        for epoch in range(1, train_cfg["epochs"] + 1):
            model.set_temperature(epoch, train_cfg["epochs"])
            rates = [group["lr"] for group in optimizer.param_groups]
            train_metrics, _ = run_epoch(model, train_loader, criterion, device, classes, optimizer, scaler, amp, train_cfg["grad_clip"])
            val_metrics, _ = run_epoch(model, val_loader, criterion, device, classes, amp_enabled=amp)
            history.append({"epoch": epoch, "learning_rates": rates, "temperature": model.selector.temperature,
                            "train": train_metrics, "val": val_metrics})
            log(format_epoch(epoch, train_cfg["epochs"], train_metrics, val_metrics))
            score = _score(val_metrics, train_cfg["selection_metric"])
            if score > best_score:
                best_score = score
                torch.save({"model_state": model.state_dict(), "config": cfg, "classes": classes, "fold": fold_number,
                            "epoch": epoch, "cv_metrics": val_metrics}, fold_dir / "best_model.pt")
            if scheduler is not None:
                scheduler.step()
            write_json(fold_dir / "history.json", history)
        checkpoint = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        cv_metrics, rows = run_epoch(model, val_loader, criterion, device, classes, amp_enabled=amp)
        oof_rows.extend(rows)
        write_json(fold_dir / "cv_metrics.json", cv_metrics)
        save_predictions(fold_dir / "cv_predictions.csv", rows, classes)
        save_history_csv(fold_dir / "history.csv", history, classes)
        training_curves(history, fold_dir, classes, cfg["evaluation"]["dpi"])
        evaluation_plots(cv_metrics, rows, classes, fold_dir / "cv_plots", cfg["evaluation"]["dpi"])
        fold_vis_cfg = {**cfg["evaluation"], "num_visualizations": min(6, cfg["evaluation"]["num_visualizations"])}
        save_prediction_images(rows, cv_metrics, fold_vis_cfg, fold_dir / "cv_visualizations", model.backbone.grid_size)
        holdout_metrics = None
        if train_cfg["cv_pool"] == "train":
            holdout_loader = build_loader(partitions["val"], cfg, cache_dir=cache_dir)
            holdout_metrics, holdout_rows = run_epoch(model, holdout_loader, criterion, device, classes, amp_enabled=amp)
            write_json(fold_dir / "holdout_val_metrics.json", holdout_metrics)
            save_predictions(fold_dir / "holdout_val_predictions.csv", holdout_rows, classes)
            log(f"Fold {fold_number} independent 15% validation: Acc {holdout_metrics['accuracy']:.4f} | Macro F1 {holdout_metrics['macro_f1']:.4f}")
        results.append({"fold": fold_number, "best_epoch": checkpoint["epoch"], "cv_metrics": cv_metrics, "holdout_metrics": holdout_metrics})
        write_json(output / "fold_results.json", results)
        del model, optimizer, scheduler, scaler, train_loader, val_loader, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    oof_metrics = classification_metrics([r["true_index"] for r in oof_rows], [r["probabilities"] for r in oof_rows], classes)
    write_json(output / "oof_metrics.json", oof_metrics)
    save_predictions(output / "oof_predictions.csv", oof_rows, classes)
    evaluation_plots(oof_metrics, oof_rows, classes, output / "oof_plots", cfg["evaluation"]["dpi"])
    summary = {key: {"mean": float(np.mean([r["cv_metrics"][key] for r in results])),
                     "std": float(np.std([r["cv_metrics"][key] for r in results], ddof=1))}
               for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")}
    selected = max(results, key=lambda r: _score(r["holdout_metrics"] or r["cv_metrics"], train_cfg["selection_metric"]))
    shutil.copy2(output / f"fold_{selected['fold']}" / "best_model.pt", output / "best_model.pt")
    summary.update({"selected_fold": selected["fold"], "selection_source": "independent_15_percent_validation" if train_cfg["cv_pool"] == "train" else "fold_validation",
                    "cv_pool": train_cfg["cv_pool"], "partition_sizes": sizes,
                    "note": "Fold standard deviations describe fold variation, not independent-seed uncertainty. OOF metrics use checkpoints selected on their fold validation sets."})
    write_json(output / "cv_summary.json", summary)
    fold_comparison(results, output, cfg["evaluation"]["dpi"])
    log(f"CV macro F1: {summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['std']:.4f} | Selected fold {selected['fold']}")
    if not skip_test:
        from testing import evaluate_run
        test_metrics = evaluate_run(output)
        log(f"Final test accuracy {test_metrics['accuracy']:.4f} | Macro F1 {test_metrics['macro_f1']:.4f}")
    log(f"Run saved: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and splits without training")
    parser.add_argument("--skip-test", action="store_true", help="Leave final test evaluation for testing.py")
    args = parser.parse_args()
    main(args.config, args.dry_run, args.skip_test)
