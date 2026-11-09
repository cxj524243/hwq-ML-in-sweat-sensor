# sweat_project — concentration classification of colorimetric sweat pads

Reference implementation for the machine-learning part of the manuscript
*"<insert paper title>"*.

Five analytes (glucose, magnesium, chloride, pH, urea) are quantified from
smartphone photographs of colorimetric pads. Each photograph is cropped to a
pre-calibrated pad rectangle, described by a 3 × 3 grid of per-cell RGB means
(27 features), augmented with additive Gaussian noise in the image domain, and
classified by an RBF-SVM (k-NN for magnesium).

This repository reproduces every number reported in the manuscript's model
section, digit for digit, from the original per-modality scripts (kept in
`legacy/`).

## Repository layout

```
config.py        single source of truth: ROI rectangles, levels, noise, classifiers
run_all.py       train and evaluate all five analytes, write results/metrics.csv
src/
  io_utils.py    Unicode-safe image reader (OpenCV cannot open non-ASCII paths)
  features.py    ROI cropping, grid descriptors, noise / blank-ROI generators
  dataset.py     stratified 7:3 split + augmentation (identical RNG order to legacy)
  model.py       SVC / k-NN training and metrics
  plots.py       confusion matrix, per-level recall, PCA, predicted-vs-reference
  train.py       command-line entry point for a single analyte
legacy/          the original per-analyte scripts, unmodified
models/          fitted scaler, label encoder and classifier for each analyte
results/         metrics.csv, per-analyte JSON metrics, figures/
docs/            manuscript method section (English), parameter tables
```

## Data

460 JPEG photographs (4000 × 3000 px), ten per calibration level, stored as one
folder per analyte (`glu/`, `Mg/`, `Cl/`, `pH/`, `urea/`) with file names
`<prefix><level>_<index>.jpg`, e.g. `glu=1.5_03.jpg`. Within each level the first
seven photographs (by acquisition index) form the training set and the remaining
three the test set.

The image set is not included in this archive because of its size (2.3 GB). It is
available from the corresponding author upon reasonable request; unpack it so
that `data/<analyte>/` is populated, or point `--data-root` / `SWEAT_DATA_ROOT`
at an existing copy.

## Requirements

Python 3.11 with `numpy==1.26.4`, `opencv-python==4.13.0.88`,
`scikit-learn==1.9.1`, `matplotlib==3.10.9`, `joblib==1.6.0`
(`pip install -r requirements.txt`).

## Usage

```bash
python run_all.py                 # all five analytes -> results/metrics.csv + figures
python -m src.train --modality glu
python run_all.py --no-figures    # numbers only
python run_all.py --no-abnormal   # disable the evaluation-set perturbation
python -m src.train --modality ph --data-root /path/to/images
```

All experimental parameters live in `config.py`; no other file needs editing.

## Reported results

| Analyte | Classifier | Classes | Train | Test | Accuracy | Macro-F1 | MAE |
|---|---|---|---|---|---|---|---|
| Glucose | RBF-SVM (C = 10) | 8 | 1176 | 504 | 99.60% | 0.996 | 0.003 mM |
| Magnesium | k-NN (k = 3) | 7 | 784 | 336 | 99.70% | 0.997 | 0.009 mM |
| Chloride | RBF-SVM (C = 10) | 9 | 1323 | 567 | 99.65% | 0.996 | 0.256 mM |
| pH | RBF-SVM (C = 10) | 11 | 1617 | 693 | 99.86% | 0.999 | 0.002 |
| Urea | RBF-SVM (C = 10) | 10 | 1470 | 630 | 99.52% | 0.995 | 0.063 mM |

These figures are obtained on the noise-augmented evaluation set described in
the method section (ten noise levels × two replicates per test photograph).
Reproducibility was verified by re-running the original scripts in `legacy/`
after changing only their data path and their image reader; both produce the
same accuracies to the sixth decimal (see `results/metrics.csv`).

## Citation

```bibtex
@article{<key>,
  title   = {<title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>}
}
```

## License

<to be decided by the authors>
