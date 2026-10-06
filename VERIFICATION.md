# Verification

Verified on Windows with Python 3.12.14 and CPU PyTorch. Exact top-level dependency versions are recorded in `requirements-tested.txt`.

- Full test suite: **14 passed** in 53.33 seconds.
- All seven YAML configurations loaded and validated.
- All Python files passed bytecode compilation.
- A standard 224x224 ViT-B/16 plus 31-patch/CLS GraphSAGE model completed an offline forward pass.
- The complete training workflow ran **five folds with twenty epochs each**, using 120 synthetic RGB images and a small custom ViT.
- Synthetic partition sizes were 84 training, 18 independent validation, and 18 test images.
- Every fold saved its checkpoint, twenty-epoch history, per-class metrics, learning curves, and prediction outputs.
- Final checkpoint selection used the independent validation partition; the saved test partition was evaluated afterward.
- Generated learning curves, confusion matrices, and actual/predicted figures were visually inspected. Prediction figures reserve space for aggregate metric headings.
- Tests also cover selector classification gradients, unique selection, temperature-dependent surrogates, correct CLS initialization, spatial node indexing, graph batch isolation, component toggles, deterministic/image-disjoint/group-disjoint splits, duplicate/corrupt-image rejection, CSV labels, binary metrics, cache reuse, and cache invalidation when frozen weights change.

One warning originates from a dependency's use of the deprecated `torch.jit.script`; it did not cause a test failure.

This verification establishes software behavior. It does not establish brain-tumor accuracy, CUDA/AMP numerical behavior, or the effectiveness of the architectural changes. Pretrained ImageNet weight downloading/loading was not exercised in the offline tests. The corrected model must be trained and evaluated on the intended real dataset before reporting scientific performance. Original notebook metrics are retained only as legacy artifacts.
