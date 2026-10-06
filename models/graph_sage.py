from torch import nn
from torch_geometric.nn import SAGEConv

from .graph_builder import build_batch_graph


def pool_tokens(tokens, mode):
    if mode == "cls":
        return tokens[:, 0]
    if mode == "mean":
        return tokens.mean(dim=1)
    import torch
    return torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, grid_size, cfg):
        super().__init__()
        self.cfg, self.grid_size = cfg, grid_size
        dim = cfg["hidden_dim"] if cfg["enabled"] else input_dim
        self.output_dim = dim * (2 if cfg["pooling"] == "mean_max" else 1)
        self.convs, self.norms, self.skips = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        if cfg["enabled"]:
            for layer in range(cfg["layers"]):
                incoming = input_dim if layer == 0 else dim
                self.convs.append(SAGEConv(incoming, dim))
                self.norms.append(nn.LayerNorm(dim))
                self.skips.append(nn.Linear(incoming, dim) if incoming != dim else nn.Identity())
        self.activation, self.dropout = nn.GELU(), nn.Dropout(cfg["dropout"])

    def forward(self, tokens, selected_indices):
        if self.cfg["enabled"]:
            x, edge_index = build_batch_graph(tokens, selected_indices, self.grid_size, self.cfg)
            for conv, norm, skip in zip(self.convs, self.norms, self.skips):
                x = self.dropout(self.activation(norm(conv(x, edge_index))) + skip(x))
            tokens = x.view(tokens.size(0), tokens.size(1), -1)
        return pool_tokens(tokens, self.cfg["pooling"])
