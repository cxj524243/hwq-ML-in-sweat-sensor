# Method section — academic English version

> Content is taken verbatim from the implemented pipeline (`config.py`, `src/`).
> Style follows the reference paragraph you provided (continuous narration,
> parameter-dense, closing reproducibility statement).
> Verified numbers: see `results/metrics.csv`; legacy equivalence: five scripts
> in `legacy/` reproduce the same accuracies to the sixth decimal.

---

## 2.x Machine-learning classification of analyte concentrations

The concentration of each analyte was inferred from smartphone photographs of the colorimetric pads by a supervised classification pipeline consisting of four stages: pad localisation, colour-descriptor construction, noise-based data augmentation, and kernel-based classification. The same feature pipeline and the same implementation were used for all five analytes; the five tasks differ only in the set of calibration levels and in the pixel coordinates of the region of interest (ROI), and only the final classifier differs for magnesium. Images were acquired with a 12-megapixel camera (4000 × 3000 pixels) under fixed illumination and a constant working distance, and each calibration level was photographed ten times. Because the lateral position of the pad is fixed by the sample holder, the colorimetric area was cropped with a pre-calibrated axis-aligned rectangle rather than being segmented by contour detection, which avoids the contour-fragmentation errors that arise at this pad size; the cropped patch was then converted from BGR to RGB to remove the camera's channel ordering. Each ROI was partitioned into a 3 × 3 grid, and the arithmetic mean of every colour channel was computed within each cell, yielding a 27-dimensional descriptor (row-major over cells, in R, G, B order). The descriptor was divided by 255 so that all components lie in [0, 1] and measurements acquired in different sessions remain on a common scale. Compared with a single whole-pad mean, the grid descriptor retains part of the spatial heterogeneity of colour development while keeping the input dimensionality at 27, which is far below that of the raw patch and makes the descriptor more robust to local inhomogeneities such as uneven reagent wetting.

Data augmentation was performed in the image domain rather than in the feature domain: independent zero-mean Gaussian noise was added to the pixel values of the ROI, the result was clipped to [0, 255], and the grid descriptor was re-extracted, so that every augmented sample passes through exactly the same pipeline as a measured one. For training, five noise levels σ ∈ {0.5, 1.0, 1.5, 2.0, 2.5} were applied four times each; for evaluation, ten levels σ ∈ {0.5, 1.0, …, 5.0} were applied twice each, which yields 21 descriptors per photograph (one unperturbed and 20 perturbed) and a test set that deliberately contains the same acquisition and sensor noise as the training data. The dataset was partitioned in a stratified manner: within each calibration level the photographs were sorted by acquisition index, the first seven were assigned to training and the remaining three to testing, giving a 7:3 split with 147 training and 63 test descriptors per level (112 and 48 for magnesium, for which three noise replicates were used). No validation split was held out; all hyperparameters were fixed a priori and left unchanged across analytes, so no model selection was performed on the evaluation data. Feature standardisation (zero mean, unit variance) was fitted on the training descriptors only and applied unchanged to the test descriptors, so that no information from the evaluation set enters the transformation.

Classification was performed with a radial-basis-function support vector machine (RBF-SVM) with penalty parameter C = 10 and kernel width γ set to the scikit-learn 'scale' convention, γ = 1/(n_features · Var(X)), using the one-versus-one scheme for the multiclass problem. For magnesium, whose calibration range extends from 0 to 8 mM with a 0.5 mM spacing at the low end, a distance-weighted k-nearest-neighbour classifier with k = 3 was used instead. The number of output classes equals the number of calibration levels of each analyte, namely 8 (glucose), 7 (magnesium), 9 (chloride), 11 (pH) and 10 (urea), and is determined from the training labels rather than being fixed by hand. Performance was quantified by the overall accuracy, the macro-averaged F1 score, the mean absolute error (MAE) in concentration units, the confusion matrix, and the recall of each calibration level; in addition, the standardised training descriptors were projected onto their first two principal components (PCA) and plotted with 95% confidence ellipses in order to assess the separability of the calibration clusters independently of the classifier. The pipeline was implemented in Python 3.11; the versions of the packages used are listed in the requirements file provided with the code. the random seed was fixed at 42 for the noise generator and the data partition, and the fitted scaler and classifier were serialised with joblib. The training and evaluation code and the fitted models are available at https://github.com/cxj524243/hwq-ML-in-sweat-sensor/releases/tag/code. Because of its size, the original image dataset is available separately upon request from the authors.

---

## Table 1. Experimental parameters of the five classification tasks

| Analyte | Classes | Calibration levels | ROI (x₁,y₁)→(x₂,y₂), px | Training noise σ (replicates) | Test noise σ (replicates) | Classifier | Train / test per level |
|---|---|---|---|---|---|---|---|
| Glucose | 8 | 0–2.0 mM, step 0.3 | (1292,1817)→(1647,2156) | {0.5,1,1.5,2,2.5} ×4 | {0.5,1,…,5.0} ×2 | RBF-SVM, C = 10, γ = 'scale' | 147 / 63 |
| Magnesium | 7 | 0, 0.5, 1, 2, 4, 5, 8 mM | (1528,1804)→(1542,1815) | {1.0,1.5,2,2.5,3.0} ×3 | {1.0,1.5,2,2.5,3.0} ×3 | kNN, k = 3, distance-weighted | 112 / 48 |
| Chloride | 9 | 0, 5, 15, 30, 45, 60, 75, 90, 100 mM | (1441,2039)→(1604,2209) | {0.5,1,1.5,2,2.5} ×4 | {0.5,1,…,5.0} ×2 | RBF-SVM, C = 10, γ = 'scale' | 147 / 63 |
| pH | 11 | 3.0–8.0, step 0.5 | (1441,2039)→(1604,2209) | {0.5,1,1.5,2,2.5} ×4 | {0.5,1,…,5.0} ×2 | RBF-SVM, C = 10, γ = 'scale' | 147 / 63 |
| Urea | 10 | 0, 5, 10, 15, 20, 25, 30, 50, 80, 100 mM | (1329,1696)→(1673,2028) | {2,3,4,5,6} ×4 | {1.0,2,…,10.0} ×2 | RBF-SVM, C = 10, γ = 'scale' | 147 / 63 |

Sample accounting: 10 photographs per level → 7 × (1 + 5σ × 4) = 147 training and 3 × (1 + 10σ × 2) = 63 test descriptors; magnesium uses 7 × (1 + 5σ × 3) = 112 and 3 × (1 + 5σ × 3) = 48.

---

## Notes — internal, not part of the manuscript

1. **The held-out set is noise-augmented.** Every sentence above that quotes an accuracy therefore describes a *noise-robustness* condition, not a clean-set accuracy. If the reviewers expect a conventional test accuracy, either add one sentence reporting the σ = 0 subset separately, or move these figures into a robustness subsection.
2. **Deliberate perturbation of the evaluation set** (in the code: 2 test descriptors for glucose and chloride, 4 for urea and 1 for pH are replaced by the descriptor of a random-colour ROI while the label is kept; in addition, one all-white ROI is inserted into the magnesium 5 mM class) is currently *not* mentioned in the English text above. For a reproducible release it must either be described or removed; the corresponding switch is `--no-abnormal` (`config.py: abnormal_count`).
3. **No Adam, no learning rate, no dropout, no batch normalisation, no 6:2:2 split, no five-channel tensor, no convolution.** Those elements belong to the reference paragraph's architecture, not to this implementation; do not let them migrate into the manuscript.
4. If an English caption is needed for the supplementary parameter table, "Table S1. Acquisition and model parameters for the five concentration-classification tasks" is consistent with the wording above.
