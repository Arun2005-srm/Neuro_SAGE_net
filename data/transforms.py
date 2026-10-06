from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transform(cfg, training=False):
    steps = [transforms.Resize((cfg["image_size"], cfg["image_size"]), interpolation=InterpolationMode.BILINEAR, antialias=True)]
    if training:
        aug = cfg["augmentation"]
        if aug["horizontal_flip"]:
            steps.append(transforms.RandomHorizontalFlip(aug["horizontal_flip"]))
        if aug["vertical_flip"]:
            steps.append(transforms.RandomVerticalFlip(aug["vertical_flip"]))
        if aug["rotation"]:
            steps.append(transforms.RandomRotation(aug["rotation"], interpolation=InterpolationMode.BILINEAR))
    steps += [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    return transforms.Compose(steps)
