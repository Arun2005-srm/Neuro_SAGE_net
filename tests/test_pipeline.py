import json
from pathlib import Path

import pytest
import yaml

from testing import evaluate_run
from train import main
from conftest import make_images


def test_full_five_fold_twenty_epoch_protocol_and_artifacts(tiny_config, tmp_path):
    tiny_config["dataset"]["roots"] = [str(make_images(tmp_path / "images"))]
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config))
    run = main(config_path)
    split = json.loads((run / "splits.json").read_text())
    assert [len(split["partitions"][name]) for name in ("train", "val", "test")] == [84, 18, 18]
    for fold in range(1, 6):
        folder = run / f"fold_{fold}"
        history = json.loads((folder / "history.json").read_text())
        assert len(history) == 20
        assert (folder / "best_model.pt").is_file()
        assert (folder / "training_curves.png").is_file()
        assert (folder / "class_accuracy_curves.png").is_file()
        assert json.loads((folder / "holdout_val_metrics.json").read_text())["samples"] == 18
    assert json.loads((run / "oof_metrics.json").read_text())["samples"] == 84
    assert json.loads((run / "test_metrics.json").read_text())["samples"] == 18
    assert (run / "test_plots" / "roc_pr_curves.png").is_file()
    assert len(list((run / "test_visualizations").glob("*_actual_predicted.png"))) == 2
    log = (run / "training.log").read_text()
    assert log.count("ep[20/20]") == 5
    assert "class acc" in log
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_run(run)
    with pytest.raises(ValueError, match="not empty"):
        main(config_path)
