"""Vectorized spatial graphs: CLS is always node 0, selected patches start at 1."""
import torch


def adjacency_matrix(indices, grid_size, cfg):
    batch, count = indices.shape
    rows, cols = indices // grid_size, indices % grid_size
    dr = (rows.unsqueeze(2) - rows.unsqueeze(1)).abs()
    dc = (cols.unsqueeze(2) - cols.unsqueeze(1)).abs()
    if cfg["connectivity"] == 8:
        spatial = (torch.maximum(dr, dc) == 1)
    else:
        spatial = (dr + dc == 1)
    if cfg["topology"] == "none":
        spatial = torch.zeros_like(spatial)
    elif cfg["topology"] == "random":
        # Reassign spatial topology to different patch features while preserving edge count.
        # CLS remains fixed; evaluation uses a fixed permutation for reproducible ablations.
        if count > 1:
            generator = torch.Generator().manual_seed(314159)
            perm = torch.randperm(count, generator=generator).to(indices.device)
            spatial = spatial[:, perm][:, :, perm]
    adjacency = torch.zeros(batch, count + 1, count + 1, dtype=torch.bool, device=indices.device)
    adjacency[:, 1:, 1:] = spatial
    if cfg["cls_hub"]:
        adjacency[:, 0, 1:] = True
        adjacency[:, 1:, 0] = True
    if cfg["self_loops"]:
        adjacency.diagonal(dim1=1, dim2=2).fill_(True)
    return adjacency


def build_batch_graph(tokens, indices, grid_size, cfg):
    adjacency = adjacency_matrix(indices, grid_size, cfg)
    graph, source, destination = adjacency.nonzero(as_tuple=True)
    nodes = tokens.size(1)
    edge_index = torch.stack([source + graph * nodes, destination + graph * nodes])
    return tokens.reshape(-1, tokens.size(-1)), edge_index
