"""Unique hard top-k selection with a straight-through soft gradient surrogate."""
import torch
import torch.nn.functional as F
from torch import nn


class TokenSelector(nn.Module):
    def __init__(self, dim, grid_size, cfg):
        super().__init__()
        self.cfg, self.grid_size = cfg, grid_size
        self.temperature = cfg["temperature_start"]
        self.score_net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cfg["hidden_dim"]), nn.GELU(),
                                       nn.Dropout(cfg["dropout"]), nn.Linear(cfg["hidden_dim"], 1)) if cfg["enabled"] and cfg["mode"] == "learned" else None

    def forward(self, tokens):
        cls, patches = tokens[:, :1], tokens[:, 1:]
        batch, count, dim = patches.shape
        if not self.cfg["enabled"]:
            indices = torch.arange(count, device=tokens.device).expand(batch, -1)
            return tokens, {"selected_indices": indices, "scores": None, "assignments": None}
        if self.cfg["mode"] == "learned":
            scores = self.score_net(patches).squeeze(-1).float()
            kernel = self.cfg["spatial_smoothing"]
            scores = F.avg_pool2d(scores.view(batch, 1, self.grid_size, self.grid_size), kernel, stride=1,
                                 padding=kernel // 2, count_include_pad=False).flatten(1)
        elif self.cfg["mode"] == "norm":
            scores = patches.float().norm(dim=-1)
        else:
            # Fixed pseudorandom ranking at evaluation, without a trained scorer.
            positions = torch.arange(count, device=tokens.device, dtype=torch.float32)
            scores = torch.rand(batch, count, device=tokens.device) if self.training else torch.frac(torch.sin(positions * 12.9898 + 78.233) * 43758.5453).expand(batch, -1)
        rank_scores = scores
        if self.training and self.cfg["mode"] == "learned" and self.cfg["gumbel_noise"]:
            uniform = torch.rand_like(scores).clamp_(1e-6, 1 - 1e-6)
            rank_scores = scores - torch.log(-torch.log(uniform))
        indices = rank_scores.topk(self.cfg["num_tokens"], dim=-1).indices
        assignments = None
        if self.training and self.cfg["mode"] == "learned":
            hard = F.one_hot(indices, num_classes=count).float()
            # Each slot excludes earlier hard choices. Forward selects distinct patches;
            # backward uses a temperature-dependent soft assignment to train the scorer.
            previously_selected = (hard.cumsum(dim=1) - hard).bool()
            soft_logits = rank_scores.unsqueeze(1).expand_as(hard).masked_fill(previously_selected, -torch.inf)
            assignments = torch.softmax(soft_logits / self.temperature, dim=-1)
            straight_through = hard + (assignments - assignments.detach())
            selected = torch.bmm(straight_through.to(patches.dtype), patches)
        else:
            selected = patches.gather(1, indices.unsqueeze(-1).expand(-1, -1, dim))
        return torch.cat([cls, selected], dim=1), {"selected_indices": indices, "scores": scores, "assignments": assignments}
