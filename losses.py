import torch
import torch.nn.functional as F
from torch import nn


class ClassificationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits, labels, aux):
        ce = F.cross_entropy(logits.float(), labels, label_smoothing=self.cfg["label_smoothing"])
        assignments = aux.get("assignments")
        overlap = logits.new_zeros((), dtype=torch.float32)
        if self.cfg["selection_overlap_weight"] and assignments is not None and assignments.size(1) > 1:
            gram = assignments @ assignments.transpose(1, 2)
            k = assignments.size(1)
            mask = ~torch.eye(k, device=logits.device, dtype=torch.bool)
            overlap = gram[:, mask].mean()
        return ce + self.cfg["selection_overlap_weight"] * overlap
