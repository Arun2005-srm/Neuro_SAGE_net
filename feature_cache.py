"""Per-image frozen-prefix caches keyed by weights, transforms, and image contents."""
import hashlib
import json
from pathlib import Path

import torch

from data.loaders import build_loader


@torch.no_grad()
def prepare_cache(model, records, cfg, device):
    if not cfg["cache"]["enabled"]:
        return None
    if not model.backbone.prefix_fully_frozen:
        raise ValueError("Cannot cache a trainable prefix")
    hasher = hashlib.sha256(json.dumps(cfg["dataset"]["transforms"], sort_keys=True).encode())
    for name, tensor in sorted(model.backbone.prefix_state().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    directory = Path(cfg["cache"]["directory"]) / hasher.hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    pending = [r for r in records if not (directory / f"{r.sample_id}.pt").exists()]
    model.eval()
    if pending:
        print(f"Caching frozen prefix: {len(pending)} images", flush=True)
    for batch in build_loader(pending, cfg) if pending else []:
        tokens = model.backbone.frozen_forward(batch["image"].to(device)).cpu().float()
        for sample_id, feature in zip(batch["sample_id"], tokens):
            target = directory / f"{sample_id}.pt"
            temporary = target.with_suffix(".tmp")
            torch.save(feature.clone(), temporary)
            temporary.replace(target)
    return directory
