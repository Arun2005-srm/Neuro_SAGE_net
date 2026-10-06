from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from config import load_config


@pytest.fixture
def tiny_config(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "datasets" / "example_folders.yaml")
    cfg["runtime"].update(device="cpu", use_amp=False, cpu_threads=1)
    cfg["model"]["backbone"].update(name="custom", pretrained=False, freeze_blocks=1,
                                      custom={"patch_size": 8, "num_layers": 3, "num_heads": 4, "hidden_dim": 32, "mlp_dim": 64})
    cfg["model"]["selector"].update(num_tokens=5, hidden_dim=16, dropout=0.0, spatial_smoothing=1)
    cfg["model"]["projector"].update(hidden_dim=24, output_dim=16, dropout=0.0)
    cfg["model"]["graph"].update(hidden_dim=16, layers=2, dropout=0.0)
    cfg["model"]["head"].update(hidden_dim=16, dropout=0.0)
    cfg["dataset"]["transforms"]["image_size"] = 32
    cfg["training"].update(output_dir=str(tmp_path / "run"), learning_rate=.001, backbone_learning_rate=.001)
    cfg["cache"]["directory"] = str(tmp_path / "cache")
    cfg["data_loader"].update(batch_size=32, num_workers=0, pin_memory=False)
    cfg["evaluation"].update(num_visualizations=2, dpi=45)
    return cfg


def make_images(root, per_class=40, classes=("alpha", "beta", "gamma")):
    root = Path(root)
    rng = np.random.default_rng(17)
    for label, name in enumerate(classes):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            image = rng.integers(0, 50, (32, 32, 3), dtype=np.uint8)
            image[:, :, label % 3] += 170
            Image.fromarray(image).save(folder / f"patient_{index:03d}_{label}.png")
    return root
