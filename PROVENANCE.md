# Refactor Sources

Original project:

- Repository: https://github.com/Arun2005-srm/NeuroSAGE-Net-Brain-Tumor-Classification
- Reviewed commit: `d2fa0b2cdf61e566fcc55d165d863be7fd65da84`
- Original source: `src/NeuroSAGE_Net.ipynb.ipynb`
- Preserved copy: `notebooks/original_neurosage.ipynb`
- Original result artifacts: `legacy_results/`

Organization reference:

- Repository: https://github.com/Arun2005-srm/skin-lesion-segmentation-refactored
- Reviewed commit: `6c7fb4899279d57994b4fa89055c49a295e0009a`
- Patterns followed: separate component files, dataset adapters, inherited YAML configurations, training/testing entry points, saved metrics and figures.

The corrected implementation changes model semantics and requires new caches and training. It does not preserve numerical equivalence with the notebook or claim its reported accuracy. No GitHub changes have been pushed by this refactor.
