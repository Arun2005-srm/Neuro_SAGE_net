import torch

from losses import ClassificationLoss
from models import NeuroSAGENet
from models.graph_builder import adjacency_matrix, build_batch_graph
from models.token_selector import TokenSelector
from runtime import setup_runtime


def test_default_vit_graph_pipeline_runs_without_downloads(tiny_config):
    from pathlib import Path
    from config import load_config
    cfg = load_config(Path(__file__).parents[1] / "configs" / "datasets" / "example_folders.yaml")
    setup_runtime({**cfg["runtime"], "device": "cpu", "cpu_threads": 2})
    model = NeuroSAGENet(cfg, 4, load_pretrained=False).eval()
    with torch.no_grad():
        logits, aux = model(images=torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)
    assert aux["selected_indices"].shape == (1, 31)
    assert model.backbone.grid_size == 14
    assert len(model.backbone.remaining) == 3


def test_classifier_loss_trains_selector_and_keeps_prefix_frozen(tiny_config):
    setup_runtime(tiny_config["runtime"])
    model = NeuroSAGENet(tiny_config, 3)
    model.train()
    logits, aux = model(images=torch.randn(4, 3, 32, 32))
    # No auxiliary loss: this must be a classification-driven gradient.
    ClassificationLoss({"label_smoothing": 0.0, "selection_overlap_weight": 0.0})(logits, torch.tensor([0, 1, 2, 1]), aux).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.selector.score_net.parameters())
    assert all(p.grad is None for p in model.backbone.prefix_parameters())
    assert not model.backbone.prefix.training
    assert any(p.grad is not None for p in model.backbone.selection_block.parameters())
    assert logits.shape == (4, 3)
    assert all(len(set(row.tolist())) == 5 for row in aux["selected_indices"])


def test_hard_forward_matches_selected_tokens_and_temperature_changes_surrogate(tiny_config):
    cfg = {**tiny_config["model"]["selector"], "gumbel_noise": False}
    selector = TokenSelector(32, 4, cfg).train()
    tokens = torch.randn(2, 17, 32)
    selected, aux = selector(tokens)
    gathered = tokens[:, 1:].gather(1, aux["selected_indices"].unsqueeze(-1).expand(-1, -1, 32))
    assert torch.allclose(selected[:, 1:], gathered)
    assert torch.equal(selected[:, 0], tokens[:, 0])
    selector.temperature = .2
    _, colder = selector(tokens)
    assert torch.equal(aux["selected_indices"], colder["selected_indices"])
    assert not torch.allclose(aux["assignments"], colder["assignments"])


def test_spatial_graph_uses_correct_cls_and_patch_offsets(tiny_config):
    cfg = {**tiny_config["model"]["graph"], "connectivity": 4, "cls_hub": True}
    indices = torch.tensor([[0, 1, 15], [4, 5, 14]])
    adjacency = adjacency_matrix(indices, 4, cfg)
    assert adjacency.shape == (2, 4, 4)
    assert adjacency[:, 0, 1:].all() and adjacency[:, 1:, 0].all()
    assert adjacency[0, 1, 2] and adjacency[0, 2, 1]
    assert not adjacency[0, 1, 3] and not adjacency[0, 2, 3]
    assert not adjacency.diagonal(dim1=1, dim2=2).any()
    tokens = torch.randn(2, 4, 16)
    _, edges = build_batch_graph(tokens, indices, 4, cfg)
    assert torch.equal(edges[0] // 4, edges[1] // 4)


def test_cls_token_is_distinct_from_position_embedding(tiny_config):
    cfg = tiny_config
    cfg["model"]["backbone"]["freeze_blocks"] = 0
    model = NeuroSAGENet(cfg, 3)
    assert torch.count_nonzero(model.backbone.cls_token) == 0
    assert torch.count_nonzero(model.backbone.position_embedding[:, :1]) > 0
    tokens = model.backbone.frozen_forward(torch.randn(2, 3, 32, 32))
    assert torch.allclose(tokens[:, :1], model.backbone.position_embedding[:, :1].expand(2, -1, -1))


def test_cached_and_image_inference_agree_and_component_toggles_work(tiny_config):
    setup_runtime(tiny_config["runtime"])
    for pooling in ("cls", "mean", "mean_max"):
        tiny_config["model"]["graph"].update(enabled=False, pooling=pooling)
        tiny_config["model"]["selector"]["enabled"] = False
        tiny_config["model"]["projector"]["enabled"] = False
        model = NeuroSAGENet(tiny_config, 2).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            from_images, _ = model(images=images)
            from_cache, _ = model(tokens=model.backbone.frozen_forward(images))
        assert from_images.shape == (2, 2)
        assert torch.allclose(from_images, from_cache, atol=1e-6)
