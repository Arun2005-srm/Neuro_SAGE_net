from torch import nn


class TokenProjector(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.output_dim = cfg["output_dim"] if cfg["enabled"] else input_dim
        self.layers = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, cfg["hidden_dim"]), nn.GELU(),
                                    nn.Dropout(cfg["dropout"]), nn.Linear(cfg["hidden_dim"], cfg["output_dim"]),
                                    nn.LayerNorm(cfg["output_dim"])) if cfg["enabled"] else nn.Identity()

    def forward(self, tokens):
        return self.layers(tokens)
