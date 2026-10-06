import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_transform


class ImageClassificationDataset(Dataset):
    def __init__(self, records, transforms_cfg, training=False):
        self.records = records
        self.transform = build_transform(transforms_cfg, training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "label": record.label, "sample_id": record.sample_id, "path": record.path}


class CachedFeatureDataset(Dataset):
    def __init__(self, records, cache_dir, cfg, training=False):
        self.records, self.cache_dir, self.cfg, self.training = records, cache_dir, cfg, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        tokens = torch.load(self.cache_dir / f"{record.sample_id}.pt", map_location="cpu", weights_only=True).float()
        if self.training:
            tokens = tokens + self.cfg["feature_noise_std"] * torch.randn_like(tokens)
            keep = torch.rand(tokens.size(0) - 1, 1) >= self.cfg["token_dropout"]
            tokens[1:] *= keep
        return {"tokens": tokens, "label": record.label, "sample_id": record.sample_id, "path": record.path}
