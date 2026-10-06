import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CachedFeatureDataset, ImageClassificationDataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_loader(records, cfg, training=False, seed=42, cache_dir=None):
    dataset = CachedFeatureDataset(records, cache_dir, cfg["cache"], training) if cache_dir is not None else ImageClassificationDataset(records, cfg["dataset"]["transforms"], training)
    loader_cfg = cfg["data_loader"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=loader_cfg["batch_size"], shuffle=training,
                      num_workers=loader_cfg["num_workers"], pin_memory=loader_cfg["pin_memory"] and torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)
