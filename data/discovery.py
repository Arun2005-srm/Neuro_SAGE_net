"""Adapt class-folder trees or CSV manifests to a single sample contract."""
import csv
import hashlib
import re
from pathlib import Path

from PIL import Image

from .contracts import SampleRecord


def discover_records(cfg):
    rows = []
    if cfg["format"] == "folders":
        if not cfg["roots"]:
            raise ValueError("Set dataset.roots to directories containing class folders")
        extensions = {e.lower() for e in cfg["extensions"]}
        for root_value in cfg["roots"]:
            root = Path(root_value)
            if not root.is_dir():
                raise FileNotFoundError(root)
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                if cfg.get("classes") and folder.name not in cfg["classes"]:
                    continue
                paths = folder.rglob("*") if cfg["recursive"] else folder.iterdir()
                for path in sorted(p for p in paths if p.is_file() and p.suffix.lower() in extensions):
                    group = None
                    if cfg.get("group_regex"):
                        match = re.search(cfg["group_regex"], path.stem)
                        if not match or match.lastindex is None:
                            raise ValueError(f"group_regex needs a capturing group matching {path}")
                        group = match.group(1)
                    rows.append((path.resolve(), folder.name, group))
    else:
        manifest = Path(cfg["manifest"] or "")
        if not manifest.is_file():
            raise FileNotFoundError("Set dataset.manifest to a CSV file")
        root = Path(cfg.get("image_root") or manifest.parent)
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = [cfg["path_column"], cfg["label_column"]]
            if cfg.get("group_column"):
                required.append(cfg["group_column"])
            if not set(required).issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest requires columns {required}")
            for row in reader:
                raw_path = row[cfg["path_column"]].strip()
                label = row[cfg["label_column"]].strip()
                if not raw_path or not label:
                    raise ValueError("Manifest image paths and labels cannot be empty")
                group = row[cfg["group_column"]].strip() if cfg.get("group_column") else None
                rows.append(((root / raw_path).resolve(), label, group))
    if not rows:
        raise ValueError("No images discovered")
    observed = sorted({label for _, label, _ in rows})
    classes = cfg.get("classes") or observed
    if len(classes) < 2 or len(set(classes)) != len(classes) or set(observed) != set(classes):
        raise ValueError("classes must contain every observed class exactly once, with at least two classes")
    records, seen_paths, seen_hashes = [], set(), {}
    for path, name, group in sorted(rows, key=lambda row: str(row[0])):
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path) in seen_paths:
            raise ValueError(f"Image occurs more than once in configured sources: {path}")
        seen_paths.add(str(path))
        if cfg["validate_images"]:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable image: {path}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            previous = seen_hashes[digest]
            if previous[1] != name or cfg["duplicates"] == "error":
                raise ValueError(f"Exact duplicate images: {previous[0]} and {path}; conflicting labels are never allowed")
            # Removing a duplicate from different known patients would silently change grouping.
            if previous[2] != group:
                raise ValueError("Duplicate images have conflicting group IDs; reconcile the manifest")
            continue
        seen_hashes[digest] = (str(path), name, group)
        records.append(SampleRecord(str(path), classes.index(name), name, digest[:24], group))
    return records, list(classes)
