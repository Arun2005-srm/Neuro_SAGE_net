import csv
import shutil

import pytest

from config import load_config, validate_config
from data.contracts import SampleRecord
from data.discovery import discover_records
from data.splitting import assert_disjoint, make_folds, split_records
from conftest import make_images


def test_stratified_splits_and_folds_are_reproducible_and_isolated(tiny_config, tmp_path):
    tiny_config["dataset"]["roots"] = [str(make_images(tmp_path / "images"))]
    records, classes = discover_records(tiny_config["dataset"])
    parts = split_records(records, tiny_config["dataset"]["split"], classes)
    assert {k: len(v) for k, v in parts.items()} == {"train": 84, "val": 18, "test": 18}
    assert parts == split_records(records, tiny_config["dataset"]["split"], classes)
    folds = make_folds(parts["train"], tiny_config["training"], classes)
    validation_ids = [r.sample_id for _, val in folds for r in val]
    assert len(validation_ids) == len(set(validation_ids)) == 84
    holdout_ids = {r.sample_id for r in parts["val"] + parts["test"]}
    assert not holdout_ids.intersection(validation_ids)
    for train, val in folds:
        assert_disjoint([train, val, parts["val"], parts["test"]])


def test_group_splits_and_folds_never_share_patients(tiny_config):
    records = [SampleRecord(f"/{c}/{g}/{s}.png", c, str(c), f"{c}-{g}-{s}", f"{c}-{g}")
               for c in range(3) for g in range(40) for s in range(2)]
    cfg = {**tiny_config["dataset"]["split"], "strategy": "group"}
    parts = split_records(records, cfg, ["0", "1", "2"])
    assert_disjoint(parts.values(), grouped=True)
    for train, val in make_folds(parts["train"], tiny_config["training"], ["0", "1", "2"], grouped=True):
        assert_disjoint([train, val, parts["val"], parts["test"]], grouped=True)


def test_manifest_infers_classes_and_preserves_group_ids(tiny_config, tmp_path):
    root = make_images(tmp_path / "images", per_class=2, classes=("alpha", "beta"))
    manifest = tmp_path / "labels.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "label", "patient_id"])
        for path in sorted(root.rglob("*.png")):
            writer.writerow([str(path.relative_to(tmp_path)), path.parent.name, path.stem])
    cfg = {**tiny_config["dataset"], "format": "csv", "manifest": str(manifest), "group_column": "patient_id"}
    records, classes = discover_records(cfg)
    assert classes == ["alpha", "beta"]
    assert all(r.group_id for r in records)


def test_duplicates_and_corrupt_images_fail_early(tiny_config, tmp_path):
    root = make_images(tmp_path / "images", per_class=2, classes=("alpha", "beta"))
    shutil.copy2(next((root / "alpha").glob("*.png")), root / "alpha" / "duplicate.png")
    tiny_config["dataset"]["roots"] = [str(root)]
    with pytest.raises(ValueError, match="duplicate"):
        discover_records(tiny_config["dataset"])
    tiny_config["dataset"]["duplicates"] = "drop"
    assert len(discover_records(tiny_config["dataset"])[0]) == 4
    (root / "beta" / "corrupt.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unreadable"):
        discover_records(tiny_config["dataset"])


def test_config_rejects_invalid_contracts(tiny_config, tmp_path):
    validate_config(tiny_config)
    tiny_config["model"]["selector"]["num_tokens"] = 1000
    with pytest.raises(ValueError, match="num_tokens"):
        validate_config(tiny_config)
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("defaults: b.yaml\n")
    b.write_text("defaults: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(a)
