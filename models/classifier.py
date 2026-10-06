from torch import nn


class ClassificationHead(nn.Sequential):
    def __init__(self, input_dim, num_classes, cfg):
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim, cfg["hidden_dim"]), nn.GELU(),
                         nn.Dropout(cfg["dropout"]), nn.Linear(cfg["hidden_dim"], num_classes))
