import numpy as np
import torch

from data.discovery import discover_records
from data.loaders import build_loader
from feature_cache import prepare_cache
from metrics import classification_metrics
from models import NeuroSAGENet
from runtime import setup_runtime
from conftest import make_images


def test_class_accuracy_is_recall_and_binary_auc_is_supported():
    metrics = classification_metrics([0, 0, 1, 1], [[.8, .2], [.4, .6], [.1, .9], [.2, .8]], ["a", "b"])
    assert metrics["accuracy"] == .75
    assert metrics["per_class"]["a"]["accuracy"] == .5
    assert metrics["per_class"]["b"]["accuracy"] == 1
    assert metrics["macro_roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_cache_reuse_and_weight_invalidation(tiny_config, tmp_path):
    setup_runtime(tiny_config["runtime"])
    tiny_config["dataset"]["roots"] = [str(make_images(tmp_path / "images", 2, ("alpha", "beta")))]
    tiny_config["cache"]["enabled"] = True
    records, classes = discover_records(tiny_config["dataset"])
    model = NeuroSAGENet(tiny_config, len(classes)).eval()
    cache = prepare_cache(model, records, tiny_config, torch.device("cpu"))
    assert len(list(cache.glob("*.pt"))) == 4
    assert cache == prepare_cache(model, records, tiny_config, torch.device("cpu"))
    image_batch = next(iter(build_loader(records, tiny_config)))
    cache_batch = next(iter(build_loader(records, tiny_config, cache_dir=cache)))
    with torch.no_grad():
        image_logits, _ = model(images=image_batch["image"])
        cached_logits, _ = model(tokens=cache_batch["tokens"])
    assert np.allclose(image_logits.numpy(), cached_logits.numpy(), atol=1e-6)
    with torch.no_grad():
        model.backbone.cls_token.add_(.1)
    assert cache != prepare_cache(model, records, tiny_config, torch.device("cpu"))
