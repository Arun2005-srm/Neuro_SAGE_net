"""Torchvision ViT with a frozen prefix and configurable selection depth."""
from contextlib import nullcontext

import torch
from torch import nn
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer, interpolate_embeddings


class ViTBackbone(nn.Module):
    def __init__(self, cfg, image_size, load_pretrained=True):
        super().__init__()
        name = cfg["name"]
        if name == "custom":
            vit = VisionTransformer(image_size=image_size, **cfg["custom"])
        else:
            builder = getattr(models, name)
            weights = models.get_model_weights(builder).DEFAULT if cfg["pretrained"] and load_pretrained else None
            vit = builder(weights=None, image_size=image_size)
            if weights is not None:
                state = weights.get_state_dict(progress=True, check_hash=True)
                state = interpolate_embeddings(image_size, vit.patch_size, state)
                vit.load_state_dict(state)
        self.hidden_dim = vit.hidden_dim
        self.patch_size = vit.patch_size
        self.grid_size = image_size // vit.patch_size
        self.image_size = image_size
        self.freeze_blocks = cfg["freeze_blocks"]
        self.conv_proj = vit.conv_proj
        self.cls_token = nn.Parameter(vit.class_token.detach().clone(), requires_grad=not cfg["freeze_cls_token"])
        self.position_embedding = nn.Parameter(vit.encoder.pos_embedding.detach().clone(), requires_grad=not cfg["freeze_position_embedding"])
        self.input_dropout = vit.encoder.dropout
        blocks = list(vit.encoder.layers.children())
        self.prefix = nn.Sequential(*blocks[:self.freeze_blocks])
        self.selection_block = blocks[self.freeze_blocks]
        self.remaining = nn.Sequential(*blocks[self.freeze_blocks + 1:])
        self.final_norm = vit.encoder.ln
        for p in self.prefix.parameters():
            p.requires_grad = False
        if cfg["freeze_patch_embedding"]:
            for p in self.conv_proj.parameters():
                p.requires_grad = False
        self.prefix_fully_frozen = all(not p.requires_grad for p in self.prefix_parameters())
        self.train(True)

    def prefix_parameters(self):
        return [*self.conv_proj.parameters(), self.cls_token, self.position_embedding, *self.prefix.parameters()]

    def prefix_state(self):
        return {name: tensor for name, tensor in self.state_dict().items()
                if name.startswith(("conv_proj.", "prefix.")) or name in ("cls_token", "position_embedding")}

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        if self.prefix_fully_frozen:
            self.conv_proj.eval()
            self.input_dropout.eval()
        return self

    def frozen_forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} images")
        context = torch.no_grad() if self.prefix_fully_frozen else nullcontext()
        with context:
            patches = self.conv_proj(images).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(images.size(0), -1, -1)
            tokens = torch.cat([cls, patches], dim=1) + self.position_embedding
            return self.prefix(self.input_dropout(tokens))

    def before_selection(self, tokens):
        return self.selection_block(tokens)

    def after_selection(self, tokens):
        return self.final_norm(self.remaining(tokens))
