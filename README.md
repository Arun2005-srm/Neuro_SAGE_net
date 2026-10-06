# NeuroSAGE-Net

A configurable PyTorch framework for single-label image classification using a pretrained Vision Transformer, learned patch selection, and spatial GraphSAGE. The dataset pipeline is independent of the model. Classes and output channels are inferred from the dataset; no brain-tumor labels are hardcoded.

This refactor follows the component/config/data separation used in [skin-lesion-segmentation-refactored](https://github.com/Arun2005-srm/skin-lesion-segmentation-refactored). The original NeuroSAGE notebook is preserved under `notebooks/`, and its old figures and reports are under `legacy_results/`. Those metrics are from the previous implementation and do not establish performance of the corrected model.

## Layout

```text
configs/
  default.yaml                    # model, training, runtime, reporting defaults
  datasets/
    example_folders.yaml          # class-directory adapter
    example_manifest.yaml         # CSV adapter, optionally with patient IDs
    brisc.yaml                    # pools original BRISC train/test and repartitions
data/
  contracts.py                    # shared sample record
  discovery.py                    # discover/validate images, labels, groups, duplicates
  splitting.py                    # 70/15/15 and five-fold partitions
  transforms.py                   # RGB transforms and optional image augmentation
  dataset.py                      # image and cached-feature datasets
  loaders.py                      # deterministic DataLoaders
models/
  backbone.py                     # pretrained ViT, frozen prefix, trainable suffix
  token_selector.py               # hard top-k, straight-through training surrogate
  projector.py                    # configurable token projection
  graph_builder.py                # vectorized spatial adjacency; CLS node is 0
  graph_sage.py                   # residual GraphSAGE and pooling
  classifier.py                   # configurable classification head
  model.py                        # compose components
config.py                         # YAML inheritance and validation
feature_cache.py                  # content/weight/transform keyed feature cache
losses.py                         # classification and optional selection-overlap loss
metrics.py                        # aggregate and per-class metrics
trainer.py                        # train/evaluate one epoch
plots.py                          # learning, confusion, class, ROC/PR, calibration plots
visualization.py                  # actual/predicted pairs and token-location overlays
reporting.py                      # JSON/CSV artifacts
runtime.py                        # devices, AMP, seeds
train.py                          # full five-fold experiment entry point
testing.py                        # evaluate selected model on saved test partition
tests/                            # architecture, data, cache, and full workflow checks
```

## Install

Python 3.10+ is supported. Use a PyTorch/Torchvision pair compatible with your CUDA version for GPU training. The default ViT downloads ImageNet weights on first use; subsequent runs reuse Torchvision's download cache. CPU works, but full-size ViT training is substantially slower.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Dependency ranges are portable; `requirements-tested.txt` records the exact versions used for CPU verification. Install matching Torch and Torchvision builds for your own hardware.

## Configure a dataset

The supported contract is **one readable 2D image, one class label, and optionally one patient/group ID**. RGB and grayscale raster images are accepted and converted to three channels. This is not a DICOM/NIfTI, 3D-volume, multilabel, or segmentation loader; convert those formats or add an adapter that produces this contract.

### Class folders

```text
my_dataset/
  class_a/
    image_001.png
  class_b/
    image_002.jpg
  class_c/
    image_003.png
```

Copy `configs/datasets/example_folders.yaml` and edit:

```yaml
defaults: ../default.yaml
dataset:
  format: folders
  roots: [/absolute/path/to/my_dataset]
  classes: null                    # infer names, sorted once across all sources
training:
  output_dir: ../../runs/my_experiment
```

To combine existing split folders, list their roots individually. Each root must directly contain class directories; do not point to the parent of `train/class_a` and `test/class_a`. Images from every listed root are pooled, then newly split. The BRISC preset intentionally uses this new protocol, so its results cannot be compared directly against the original official test split.

### CSV labels and patient IDs

```csv
image_path,label,patient_id
images/case_001.png,class_a,patient_001
images/case_002.png,class_a,patient_001
images/case_003.png,class_b,patient_002
```

```yaml
defaults: ../default.yaml
dataset:
  format: csv
  manifest: /absolute/path/to/labels.csv
  image_root: null                 # defaults to CSV directory
  path_column: image_path
  label_column: label
  group_column: patient_id
  split:
    strategy: group
training:
  output_dir: ../../runs/patient_split
```

Column names are configurable. Absolute image paths work too. Folder datasets can extract a group ID from the first capture group in `group_regex`, for example `'^patient_(\d+)_'`; these IDs must be globally unique across all listed roots.

All configured filesystem paths are resolved relative to the **entry YAML file**, including paths inherited from defaults. CSV image paths are instead relative to `image_root` or the CSV location. Use absolute paths or forward slashes for portable Windows YAML files.

Unreadable images, repeated paths, conflicting class mappings, and exact duplicates fail before training. Set `dataset.duplicates: drop` to discard exact duplicates with matching labels and group IDs. This audits byte-identical files only; it does not establish absence of near-duplicates or patient overlap when IDs are unavailable. Missing or corrupted images are never replaced with blank tensors.

## Splitting and validation protocol

1. Discover all images and establish a single class mapping.
2. Create reproducible, stratified **70% training / 15% validation / 15% test** partitions. Integer rounding applies to image counts.
3. Run **five folds within the 70% training partition**. Each fold trains on roughly 56% of the complete dataset and validates on roughly 14%.
4. Train **all 20 epochs per fold**, validating after every epoch. There is no early stopping. Save the epoch with the best fold-validation macro F1 (configurable to accuracy, balanced accuracy, or loss).
5. After each fold, evaluate its best checkpoint on the independent 15% validation partition.
6. Select the final fold checkpoint using that 15% validation result. Evaluate the untouched 15% test partition once after all five folds.

Both partition memberships and CV memberships are saved in `splits.json`. With patient IDs, all images from one patient remain together in both holdouts and folds. Image counts may differ from exact 70/15/15 when groups have unequal sizes; patient integrity takes precedence. Partitions must contain every class, and the CV pool needs at least five samples (or five groups) represented in each class. Group holdouts choose the most class-balanced of 256 reproducible candidate assignments; at least seven groups must be represented in each class, and some group compositions need more. Errors explain insufficient data rather than silently falling back to leaking splits.

`training.cv_pool: development` is an alternative conventional protocol: pool the 70% and 15% development partitions for CV (about 68% total training / 17% fold validation). This removes the independent 15% selection holdout; selection uses fold-validation scores instead. The default is `train`, matching the strict partition interpretation above.

The 15% validation partition is used for model selection, so its metric is not an unbiased final performance estimate. Fold standard deviations are fold variation, not independent-seed confidence intervals. Aggregated out-of-fold predictions use fold checkpoints selected on those same fold-validation sets; report them as development results. The untouched test result is the final evaluation.

## Train

Validate the adapter and splits first:

```bash
python train.py --config configs/datasets/my_dataset.yaml --dry-run
```

Run the complete experiment:

```bash
python train.py --config configs/datasets/my_dataset.yaml
```

Each epoch prints aggregate train/validation metrics and a row for every class:

```text
Fold [1/5] | Train images ... | Fold validation images ...
ep[1/20] Train: Acc ... / Loss ... | Precision ... | Recall ... | F1 ... | BalancedAcc ... || Val: ...
  class_a: Train class acc ..., P ..., R ..., F1 ... | Val class acc ..., P ..., R ..., F1 ... (n=...)
```

Precision, recall, and F1 in the main epoch line are macro averages. **Class accuracy means recall among images whose actual label is that class**, rather than one-vs-rest accuracy dominated by other classes. Specificity and one-vs-rest accuracy are separately available in JSON. Training metrics are collected during optimization on training inputs, which may be augmented; validation metrics use the checkpoint in evaluation mode.

The output directory must be empty to prevent accidental overwrite. Use a new `training.output_dir` for each experiment. This release saves best inference checkpoints and histories; interrupted runs are preserved but optimizer-state resume is not implemented.

To postpone final testing:

```bash
python train.py --config configs/datasets/my_dataset.yaml --skip-test
python testing.py --run-dir runs/my_experiment
```

`testing.py` loads the saved configuration, class mapping, split, and selected checkpoint; it does not rediscover or resplit the dataset. It checks test-image contents against the saved split. Repeated test evaluation requires explicit `--repeat`, intended for reproducibility verification rather than tuning.

## Architecture and configurable components

```text
Image -> patch embedding + correct pretrained CLS + position embedding
      -> frozen prefix (default blocks 1-8)
      -> one trainable selection block (default block 9)
      -> keep K distinct patches plus CLS (default K=31)
      -> remaining trainable ViT blocks (default 10-12) + final normalization
      -> optional MLP projector
      -> optional residual GraphSAGE
      -> CLS, mean, or mean+max pooling
      -> classification head with inferred class count
```

- Backbone: `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, or an unpretrained `custom` Torchvision ViT for small experiments. Input size must be divisible by patch size. Pretrained position embeddings are interpolated when input size changes.
- Selection depth: `freeze_blocks` controls the frozen prefix; selection occurs immediately after the next trainable block. All remaining blocks execute.
- Selector: enable/disable; `learned`, feature `norm`, or `random`; configurable K, scorer width, smoothing, noise, dropout, and temperatures. Random evaluation uses a fixed reproducible random ranking.
- Projector: enable/disable, hidden/output widths, dropout.
- Graph: enable/disable, 4/8 connectivity, spatial/random/none adjacency, CLS hub, layers, widths, dropout, and pooling.
- Head: hidden width/dropout. Number of outputs always comes from the dataset.

The corrected selector uses distinct hard top-k patches in the forward pass and sequential soft assignments as a **straight-through gradient surrogate** during training. Temperature changes the soft surrogate and its gradients; it does not change hard rankings for an identical noise draw. This is a biased gradient estimator, not exact differentiation through top-k. The classifier now trains the score network even with auxiliary regularization disabled. Inference uses deterministic hard selection for the learned scorer.

Graph node 0 always holds CLS. Patch node `i+1` is mapped to selected patch index `i`; adjacency uses those original grid positions. The default graph has no explicit self-loops because `SAGEConv` already transforms root-node features. Bidirectional CLS edges remain optional. Graph building is vectorized and has no unbounded topology cache.

The old probability-variance diversity term is removed. `loss.selection_overlap_weight` optionally penalizes overlap between soft selection assignments; the default is zero so the baseline learns directly from classification.

Useful ablations:

```yaml
# Plain ViT CLS classifier
model:
  selector: {enabled: false}
  projector: {enabled: false}
  graph: {enabled: false, pooling: cls}
```

```yaml
# Same selected tokens and projector, without graph reasoning
model:
  graph: {enabled: false, pooling: mean_max}
```

```yaml
# Reassign spatial edges while retaining node features and the CLS hub
model:
  graph: {topology: random}
```

Use a complete dataset config with `defaults`; the loader recursively merges these component overrides. Capacity differences still need to be considered when interpreting comparisons.

Ready-to-run BRISC variants are in `configs/ablations/plain_vit.yaml`, `no_graph.yaml`, and `random_graph.yaml`. For another dataset, change their `defaults` to your dataset YAML and set separate output directories.

## Optional caching

Image-based training is the default. Enable caching only with a fully frozen prefix:

```yaml
cache:
  enabled: true
  directory: /absolute/path/to/feature_cache
  feature_noise_std: 0.01
  token_dropout: 0.05
```

Caches are keyed by frozen parameter values and transforms, with image-content IDs for each sample. A changed prefix creates a new cache. Trainable suffixes operate on cached prefix outputs; optional feature noise/dropout affect only fold training. Image augmentation and cached features are incompatible and rejected together. Frozen modules stay in evaluation mode even when the rest of the model trains.

Test features are first extracted at final evaluation. Cache mode reports evaluation throughput from cached features, explicitly labeled as such; this must not be claimed as full image-to-prediction throughput. Prefer the default image mode for complete inference timing. Cache extraction uses float32 so cached and live prefix computation agree.

## Saved results

```text
run/
  resolved_config.yaml, environment.json, class_mapping.json
  splits.json, training.log
  fold_1/ ... fold_5/
    best_model.pt
    history.json, history.csv
    training_curves.png, class_accuracy_curves.png
    cv_metrics.json, cv_predictions.csv
    cv_plots/, cv_visualizations/
    holdout_val_metrics.json, holdout_val_predictions.csv
  best_model.pt                     # checkpoint selected without test access
  fold_results.json, cv_summary.json, fold_comparison.png
  oof_metrics.json, oof_predictions.csv, oof_plots/
  test_metrics.json, test_predictions.json, test_predictions.csv
  test_plots/
    confusion_matrix.png            # counts and normalized confusion
    class_metrics.png
    roc_pr_curves.png
    calibration.png
  test_visualizations/
    *_actual_predicted.png
    *_selected_patches.png
```

Each actual/predicted pair shows the original image, true class, predicted class, confidence, correctness, and aggregate evaluation accuracy/F1. Per-image F1 is not reported because it is not a useful classification statistic for a single example. Correct and incorrect cases are sampled reproducibly when available. Each fold saves up to six validation examples; final test uses `evaluation.num_visualizations`.

Selected-patch overlays show retained token locations, not segmentation or validated explanations. Late ViT tokens and CLS already contain global information, so lesion overlap does not prove that predictions depend exclusively on the highlighted regions.

## Verification and performance

The test suite verifies selector classification gradients, CLS initialization, unique selections, graph offsets/batch isolation, component toggles, deterministic splits, patient isolation, duplicate/image validation, binary metrics, cache invalidation, and the complete five-fold/twenty-epoch workflow including generated figures. The full workflow test uses synthetic RGB images and a small custom ViT, with no pretrained downloads. It verifies software behavior and does not measure brain-tumor performance.

Actual BRISC training requires the dataset and a suitable GPU environment. The original notebook's checkpoints are incompatible with the corrected model. Regenerate caches and retrain before reporting results.
