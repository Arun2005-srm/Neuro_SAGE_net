"""70/15/15 holdouts and five-fold CV, optionally keeping patient groups intact."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split


def _check(records, classes, description, grouped=False):
    if not records or set(r.label for r in records) != set(range(len(classes))):
        raise ValueError(f"{description} must contain every class; increase dataset size or adjust group composition")
    if grouped and any(not r.group_id for r in records):
        raise ValueError("Group splitting requires a group ID for every image")


def assert_disjoint(partitions, grouped=False):
    paths, groups = set(), set()
    for records in partitions:
        current = {r.path for r in records}
        if paths & current:
            raise ValueError("Image overlap between partitions")
        paths |= current
        if grouped:
            current_groups = {r.group_id for r in records}
            if groups & current_groups:
                raise ValueError("Group overlap between partitions")
            groups |= current_groups


def split_records(records, cfg, classes):
    grouped = cfg["strategy"] == "group"
    _check(records, classes, "Dataset", grouped)
    labels = np.array([r.label for r in records])
    indices = np.arange(len(records))
    seed = cfg["seed"]
    if grouped:
        # Choose the most balanced reproducible group assignment. Exact image ratios
        # may be impossible with unequal groups; patient integrity takes precedence.
        groups = np.array([r.group_id for r in records])
        counts = [len({r.group_id for r in records if r.label == c}) for c in range(len(classes))]
        if min(counts) < 7:
            raise ValueError("Group splitting plus five-fold CV needs at least seven groups represented in each class")
        splitter = GroupShuffleSplit(n_splits=256, train_size=.7, random_state=seed)
        totals = np.bincount(labels, minlength=len(classes))
        best = None
        for attempt, (candidate_train, held) in enumerate(splitter.split(indices, labels, groups)):
            inner = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed + attempt)
            val_local, test_local = next(inner.split(held, labels[held], groups[held]))
            candidate_val, candidate_test = held[val_local], held[test_local]
            candidate = (candidate_train, candidate_val, candidate_test)
            if any(set(labels[part]) != set(range(len(classes))) for part in candidate):
                continue
            if any(len(set(groups[candidate_train][labels[candidate_train] == c])) < 5 for c in range(len(classes))):
                continue
            score = sum(float(np.square(np.bincount(labels[part], minlength=len(classes)) / totals - ratio).sum())
                        for part, ratio in zip(candidate, (.7, .15, .15)))
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is None:
            raise ValueError("Cannot form group-isolated 70/15/15 partitions with every class and five CV groups per class; add groups")
        train, val, test = best[1]
    else:
        try:
            train, held = train_test_split(indices, test_size=.30, stratify=labels, random_state=seed)
            val, test = train_test_split(held, test_size=.50, stratify=labels[held], random_state=seed)
        except ValueError as exc:
            raise ValueError(f"Not enough samples for stratified 70/15/15 partitions: {exc}") from exc
    result = {name: [records[int(i)] for i in sorted(idx)] for name, idx in zip(("train", "val", "test"), (train, val, test))}
    for name, part in result.items():
        _check(part, classes, name, grouped)
    assert_disjoint(result.values(), grouped)
    return result


def make_folds(records, cfg, classes, grouped=False):
    counts = Counter(r.label for r in records)
    if min(counts.values()) < cfg["folds"]:
        raise ValueError("CV requires at least five training-pool images per class")
    labels = [r.label for r in records]
    splitter = StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg.get("seed", 42)) if grouped else StratifiedKFold(cfg["folds"], shuffle=True, random_state=cfg.get("seed", 42))
    groups = [r.group_id for r in records] if grouped else None
    if grouped and any(len({r.group_id for r in records if r.label == c}) < cfg["folds"] for c in range(len(classes))):
        raise ValueError("CV requires at least five training-pool groups per class")
    folds = []
    for train_idx, val_idx in splitter.split(np.arange(len(records)), labels, groups):
        train, val = [[records[int(i)] for i in idx] for idx in (train_idx, val_idx)]
        _check(train, classes, "Fold training", grouped)
        _check(val, classes, "Fold validation", grouped)
        assert_disjoint([train, val], grouped)
        folds.append((train, val))
    return folds


def save_splits(path, partitions, folds, classes, cv_pool):
    payload = {"classes": classes, "cv_pool": cv_pool,
               "partitions": {name: [r.to_dict() for r in records] for name, records in partitions.items()},
               "folds": [{"train_ids": [r.sample_id for r in train], "val_ids": [r.sample_id for r in val]} for train, val in folds]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
