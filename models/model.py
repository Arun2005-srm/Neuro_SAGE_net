from torch import nn

from .backbone import ViTBackbone
from .classifier import ClassificationHead
from .graph_sage import GraphSAGEEncoder
from .projector import TokenProjector
from .token_selector import TokenSelector


class NeuroSAGENet(nn.Module):
    def __init__(self, cfg, num_classes, load_pretrained=True):
        super().__init__()
        model_cfg = cfg["model"]
        self.backbone = ViTBackbone(model_cfg["backbone"], cfg["dataset"]["transforms"]["image_size"], load_pretrained)
        self.selector = TokenSelector(self.backbone.hidden_dim, self.backbone.grid_size, model_cfg["selector"])
        self.projector = TokenProjector(self.backbone.hidden_dim, model_cfg["projector"])
        self.graph = GraphSAGEEncoder(self.projector.output_dim, self.backbone.grid_size, model_cfg["graph"])
        self.classifier = ClassificationHead(self.graph.output_dim, num_classes, model_cfg["head"])

    def forward(self, images=None, tokens=None):
        if (images is None) == (tokens is None):
            raise ValueError("Provide exactly one of images or cached tokens")
        if tokens is None:
            tokens = self.backbone.frozen_forward(images)
        tokens = self.backbone.before_selection(tokens)
        tokens, aux = self.selector(tokens)
        tokens = self.backbone.after_selection(tokens)
        tokens = self.projector(tokens)
        pooled = self.graph(tokens, aux["selected_indices"])
        return self.classifier(pooled), aux

    def set_temperature(self, epoch, epochs):
        cfg = self.selector.cfg
        fraction = (epoch - 1) / max(epochs - 1, 1)
        self.selector.temperature = cfg["temperature_start"] + fraction * (cfg["temperature_end"] - cfg["temperature_start"])
