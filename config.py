"""YAML inheritance, path resolution, and configuration validation."""
from copy import deepcopy
from pathlib import Path

import yaml


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _read(path, seen):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular configuration defaults: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError("Configuration must be a YAML mapping")
    defaults = content.pop("defaults", None)
    if defaults:
        content = merge(_read(path.parent / defaults, seen | {path}), content)
    return content


def validate_config(cfg):
    for section in ("runtime", "data_loader", "dataset", "model", "training", "loss", "cache", "evaluation"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    ds, train, model = cfg["dataset"], cfg["training"], cfg["model"]
    if ds["format"] not in ("folders", "csv"):
        raise ValueError("dataset.format must be folders or csv")
    ratios = ds["split"]["ratios"]
    if set(ratios) != {"train", "val", "test"} or any(abs(ratios[k] - v) > 1e-8 for k, v in {"train": .7, "val": .15, "test": .15}.items()):
        raise ValueError("This protocol requires train/val/test ratios of 70/15/15")
    if ds["split"]["strategy"] not in ("stratified", "group"):
        raise ValueError("split.strategy must be stratified or group")
    if ds["duplicates"] not in ("error", "drop"):
        raise ValueError("dataset.duplicates must be error or drop")
    if train["cv_pool"] not in ("train", "development"):
        raise ValueError("training.cv_pool must be train or development")
    if train["folds"] != 5 or train["epochs"] != 20:
        raise ValueError("The training protocol requires 5 folds and 20 epochs per fold")
    if train["selection_metric"] not in ("macro_f1", "accuracy", "balanced_accuracy", "loss"):
        raise ValueError("Unsupported selection_metric")
    if train["scheduler"] not in ("cosine", "none"):
        raise ValueError("scheduler must be cosine or none")
    if train["learning_rate"] <= 0 or train["backbone_learning_rate"] <= 0 or train["grad_clip"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if cfg["data_loader"]["batch_size"] < 1 or cfg["data_loader"]["num_workers"] < 0:
        raise ValueError("Invalid data loader settings")
    size = ds["transforms"]["image_size"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError("image_size must be a positive integer (square input)")
    if len(ds["transforms"]["mean"]) != 3 or len(ds["transforms"]["std"]) != 3 or min(ds["transforms"]["std"]) <= 0:
        raise ValueError("RGB normalization requires three means and positive standard deviations")
    aug = ds["transforms"]["augmentation"]
    if any(not 0 <= aug[k] <= 1 for k in ("horizontal_flip", "vertical_flip")) or aug["rotation"] < 0:
        raise ValueError("Invalid image augmentation")
    backbone = model["backbone"]
    patches = {"vit_b_16": 16, "vit_b_32": 32, "vit_l_16": 16, "vit_l_32": 32}
    if backbone["name"] == "custom":
        if backbone["pretrained"]:
            raise ValueError("custom backbone cannot use pretrained weights")
        custom = backbone["custom"]
        patch_size, depth = custom["patch_size"], custom["num_layers"]
        if custom["hidden_dim"] % custom["num_heads"]:
            raise ValueError("hidden_dim must be divisible by num_heads")
    elif backbone["name"] in patches:
        patch_size = patches[backbone["name"]]
        depth = 12 if backbone["name"].startswith("vit_b") else 24
    else:
        raise ValueError("Unsupported backbone; use vit_b_16, vit_b_32, vit_l_16, vit_l_32, or custom")
    if patch_size < 1 or size % patch_size or not 0 <= backbone["freeze_blocks"] < depth:
        raise ValueError("image_size must divide into patches and freeze_blocks must leave a trainable block")
    sel = model["selector"]
    if sel["mode"] not in ("learned", "random", "norm"):
        raise ValueError("selector.mode must be learned, random, or norm")
    if not 1 <= sel["num_tokens"] <= (size // patch_size) ** 2:
        raise ValueError("num_tokens must fit the configured patch grid")
    if min(sel["temperature_start"], sel["temperature_end"]) <= 0:
        raise ValueError("Selector temperatures must be positive")
    if sel["spatial_smoothing"] < 1 or sel["spatial_smoothing"] % 2 != 1:
        raise ValueError("spatial_smoothing must be a positive odd integer")
    graph = model["graph"]
    if graph["topology"] not in ("grid", "random", "none") or graph["connectivity"] not in (4, 8):
        raise ValueError("Invalid graph topology/connectivity")
    if graph["layers"] < 1 or graph["pooling"] not in ("mean", "mean_max", "cls"):
        raise ValueError("Invalid graph layers/pooling")
    for component in (sel, model["projector"], graph, model["head"]):
        if not 0 <= component["dropout"] < 1:
            raise ValueError("Dropout must be in [0, 1)")
        for key in ("hidden_dim", "output_dim"):
            if key in component and component[key] < 1:
                raise ValueError(f"{key} must be positive")
    if not 0 <= cfg["loss"]["label_smoothing"] < 1 or cfg["loss"]["selection_overlap_weight"] < 0:
        raise ValueError("Invalid loss settings")
    cache = cfg["cache"]
    if cache["enabled"]:
        if not all(backbone[k] for k in ("freeze_patch_embedding", "freeze_cls_token", "freeze_position_embedding")):
            raise ValueError("Caching requires a fully frozen prefix, including patch/CLS/position embeddings")
        if any(aug.values()):
            raise ValueError("Image augmentation cannot be used with cached features; disable caching or image augmentation")
    if cache["feature_noise_std"] < 0 or not 0 <= cache["token_dropout"] < 1:
        raise ValueError("Invalid cache augmentation")
    if cfg["evaluation"]["num_visualizations"] < 0:
        raise ValueError("num_visualizations cannot be negative")


def load_config(path):
    path = Path(path).resolve()
    cfg = _read(path, set())
    # All user-facing paths use the entry YAML's directory, including inherited paths.
    def resolve(value):
        return str((path.parent / value).resolve()) if value else value
    ds = cfg["dataset"]
    ds["roots"] = [resolve(p) for p in ds.get("roots", [])]
    for key in ("manifest", "image_root"):
        if ds.get(key):
            ds[key] = resolve(ds[key])
    cfg["training"]["output_dir"] = resolve(cfg["training"]["output_dir"])
    cfg["cache"]["directory"] = resolve(cfg["cache"]["directory"])
    validate_config(cfg)
    return cfg
