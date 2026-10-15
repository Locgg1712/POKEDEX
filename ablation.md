# Classical ML ablation

The validation protocol is fixed for all rows: `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=42)`, first fold, with the original image ID as the
group. Augmented copies of an original image therefore cannot cross the split.
Run `kaggle/notebook_a_features.py` and then
`kaggle/notebook_b_train.py` on Kaggle to fill the measured values below.

| Experiment | Feature/model change | Val accuracy | Top-3 | Macro F1 |
|---|---|---:|---:|---:|
| Baseline | HOG + HSV histogram + Fourier (2,820) | pending | pending | pending |
| + spatial color | Baseline + local H-S histograms (3,108) | pending | pending | pending |
| + moments | Previous + Lab moments | pending | pending | pending |
| + LBP | All six groups (3,157) | pending | pending | pending |
| Resolution 128 | Separate run with 128x128 preprocessing and ppc=16 | pending | pending | pending |
| Group weighting | `1/sqrt(group dimension)` after scaling | pending | pending | pending |
| Ensemble | Five MLP seeds | pending | pending | pending |

Horizontal-flip TTA is applied at local inference time by `src/predict.py`;
it is not included as a separate offline ablation because `features.npz` does
not retain the raw images needed to construct the flipped validation features.
The exact measured output, hardest classes, confusion pairs, confidence
thresholds, `confusion.png`, and `confidence.png` are written by notebook B.
Do not claim an improvement until the corresponding row is copied from the
Kaggle output. A change that does not improve accuracy on this fixed split
should be removed or explicitly marked as rejected.

## Evaluation note

The current training loop selects the best checkpoint using validation
accuracy and then reports that same validation fold. These numbers are useful
for comparing ablations, but are optimistic as a final generalization
estimate. For a final report, reserve a second grouped fold as an untouched
test set after the ablation choice is frozen.

## Reproducibility checklist

- [ ] Notebook A saves `X`, `y`, `groups`, `names`, and `group_columns`.
- [ ] Notebook B reports support for all 150 classes (no zero-support class).
- [ ] The baseline and every ablation use the same first grouped fold.
- [ ] `Model/` contains `mlp_0.pt` ... `mlp_4.pt`, `scaler.pkl`, `labels.pkl`,
  `meta.pkl`, `centroids.pkl`, `confusion.png`, `confidence.png`, and
  `report.txt`.
- [ ] Extract `Model.zip` and run `python -m src.predict` with a sample image.
- [ ] Run `python -m src.app` and verify the GUI displays name, confidence, and
  explanation.
