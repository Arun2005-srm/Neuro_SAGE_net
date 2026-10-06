"""Actual/predicted image pairs and selected-patch location overlays."""
from pathlib import Path
import os
import random
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def save_prediction_images(rows, metrics, cfg, directory, grid_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    limit = min(cfg["num_visualizations"], len(rows))
    if not limit:
        return
    rng = random.Random(42)
    failures, successes = [[row for row in rows if row["correct"] == correct] for correct in (False, True)]
    rng.shuffle(failures)
    rng.shuffle(successes)
    # Include errors and correct predictions when both exist.
    selected = failures[:(limit + 1) // 2] + successes[:limit - min(len(failures), (limit + 1) // 2)]
    selected_ids = {r["sample_id"] for r in selected}
    selected += [r for r in failures + successes if r["sample_id"] not in selected_ids][:limit - len(selected)]
    for row in selected:
        with Image.open(row["path"]) as original:
            image = original.convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for axis in axes:
            axis.imshow(image)
            axis.axis("off")
        axes[0].set_title(textwrap.fill(f"Actual: {row['true_class']}", 36))
        axes[1].set_title(textwrap.fill(f"Predicted: {row['pred_class']}", 36) + f"\nConfidence: {row['confidence']:.1%} | {'Correct' if row['correct'] else 'Incorrect'}",
                          color="green" if row["correct"] else "red")
        fig.suptitle(f"Evaluation accuracy: {metrics['accuracy']:.2%} | Macro F1: {metrics['macro_f1']:.4f}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .90))
        fig.savefig(directory / f"{row['sample_id']}_actual_predicted.png", dpi=cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
        if cfg["save_token_overlays"]:
            fig, axis = plt.subplots(figsize=(6, 6))
            axis.imshow(image)
            width, height = image.size
            for index in row["selected_indices"]:
                y, x = divmod(index, grid_size)
                axis.add_patch(Rectangle((x * width / grid_size, y * height / grid_size), width / grid_size, height / grid_size,
                                         facecolor="#00aa88", edgecolor="white", alpha=.3))
            axis.set_title("Selected patch locations\n" + textwrap.fill(f"Actual: {row['true_class']} | Predicted: {row['pred_class']}", 40))
            axis.axis("off")
            fig.tight_layout()
            fig.savefig(directory / f"{row['sample_id']}_selected_patches.png", dpi=cfg["dpi"], bbox_inches="tight")
            plt.close(fig)
