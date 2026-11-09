"""Dataset construction: loading, stratified 7:3 split and noise augmentation.

Random-number consumption order is deliberately identical to the original
scripts so that re-running reproduces the published numbers bit for bit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from src.features import add_gaussian_noise, crop_roi, grid_features, random_noise_roi, white_roi
from src.io_utils import imread_unicode


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    counts: dict[float, tuple[int, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def _match_class(value: float, values: tuple[float, ...]) -> float | None:
    for c in values:
        if abs(value - c) < 1e-9:
            return float(c)
    return None


def load_images(mod: config.Modality, data_root: Path | None = None) -> dict[float, list[dict]]:
    """Load every image once, keyed by concentration, sorted by shooting index."""
    root = Path(data_root) if data_root else config.DATA_ROOT
    folder = root / mod.data_subdir
    if not folder.exists():
        raise FileNotFoundError(f"image folder not found: {folder}")

    buckets: dict[float, list[dict]] = {float(v): [] for v in mod.values}
    unreadable = 0
    skipped = 0

    for path in sorted(folder.glob(f"{mod.file_prefix}*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        try:
            value_txt = stem.split("_")[0].split("=", 1)[1]
            index = int(stem.split("_")[1])
            value = float(value_txt)
        except (IndexError, ValueError):
            skipped += 1
            continue

        cls = _match_class(value, mod.values)
        if cls is None:
            skipped += 1
            continue

        img = imread_unicode(path)
        if img is None:
            unreadable += 1
            continue

        buckets[cls].append(
            {"roi": crop_roi(img, mod.roi), "index": index, "name": path.name}
        )

    for cls in buckets:
        buckets[cls].sort(key=lambda item: item["index"])

    return buckets


def build_dataset(
    mod: config.Modality,
    data_root: Path | None = None,
    inject_abnormal: bool | None = None,
    seed: int = config.SEED,
    verbose: bool = True,
) -> Dataset:
    """Build the augmented train/test tensors exactly as the legacy scripts did."""
    if inject_abnormal is None:
        inject_abnormal = mod.inject_abnormal

    np.random.seed(seed)
    random.seed(seed)

    buckets = load_images(mod, data_root)
    roi_h = mod.roi[3] - mod.roi[1]
    roi_w = mod.roi[2] - mod.roi[0]

    X_train: list[np.ndarray] = []
    y_train: list[float] = []
    X_test: list[np.ndarray] = []
    y_test: list[float] = []
    counts: dict[float, tuple[int, int]] = {}
    notes: list[str] = []

    for cls in mod.values:
        cls = float(cls)
        images = buckets.get(cls, [])
        if not images:
            notes.append(f"class {cls}: no images found, skipped")
            continue

        train_imgs = images[: config.TRAIN_PER_CONC]
        test_imgs = images[
            config.TRAIN_PER_CONC : config.TRAIN_PER_CONC + config.TEST_PER_CONC
        ]
        if len(train_imgs) < config.TRAIN_PER_CONC or len(test_imgs) < config.TEST_PER_CONC:
            notes.append(
                f"class {cls}: only {len(images)} images "
                f"(need {config.TRAIN_PER_CONC + config.TEST_PER_CONC})"
            )

        for item in train_imgs:
            roi = item["roi"]
            X_train.append(grid_features(roi, config.GRID_SIZE, config.NORMALISE_TO_UNIT))
            y_train.append(cls)
            for sigma in mod.train_sigmas:
                for _ in range(mod.train_reps):
                    noisy = add_gaussian_noise(roi, sigma)
                    X_train.append(
                        grid_features(noisy, config.GRID_SIZE, config.NORMALISE_TO_UNIT)
                    )
                    y_train.append(cls)

        if mod.blank_at is not None and abs(cls - float(mod.blank_at)) < 1e-9:
            # Magnesium protocol: prepend one blank white pad, drop the first
            # augmented sample of the class, keep everything else.
            samples: list[np.ndarray] = []
            labels: list[float] = []
            for item in test_imgs:
                roi = item["roi"]
                samples.append(grid_features(roi, config.GRID_SIZE, config.NORMALISE_TO_UNIT))
                labels.append(cls)
                for sigma in mod.test_sigmas:
                    for _ in range(mod.test_reps):
                        noisy = add_gaussian_noise(roi, sigma)
                        samples.append(
                            grid_features(noisy, config.GRID_SIZE, config.NORMALISE_TO_UNIT)
                        )
                        labels.append(cls)
            blank = white_roi(roi_h, roi_w)
            X_test.append(grid_features(blank, config.GRID_SIZE, config.NORMALISE_TO_UNIT))
            y_test.append(cls)
            X_test.extend(samples[1:])
            y_test.extend(labels[1:])
            counts[cls] = (len(train_imgs), len(samples))
            continue

        for item in test_imgs:
            roi = item["roi"]
            X_test.append(grid_features(roi, config.GRID_SIZE, config.NORMALISE_TO_UNIT))
            y_test.append(cls)
            for sigma in mod.test_sigmas:
                for _ in range(mod.test_reps):
                    noisy = add_gaussian_noise(roi, sigma)
                    X_test.append(
                        grid_features(noisy, config.GRID_SIZE, config.NORMALISE_TO_UNIT)
                    )
                    y_test.append(cls)

        counts[cls] = (len(train_imgs), len(test_imgs))

    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.float64)

    n_abnormal = int(getattr(mod, 'abnormal_count', 0))
    if n_abnormal > 0:
        n_abnormal = min(n_abnormal, max(len(X_test_arr) - 1, 0))
    if inject_abnormal and n_abnormal > 0:
        # Stress test kept from the original protocol: two test rows are replaced
        # by the features of a random-colour ROI while keeping their label.
        indices = random.sample(range(len(X_test_arr)), n_abnormal)
        abnormal = grid_features(
            random_noise_roi(roi_h, roi_w), config.GRID_SIZE, config.NORMALISE_TO_UNIT
        )
        for idx in indices:
            X_test_arr[idx] = abnormal
        notes.append(f"injected abnormal ROI at test rows {sorted(indices)}")

    dataset = Dataset(
        X_train=np.asarray(X_train, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.float64),
        X_test=X_test_arr,
        y_test=y_test_arr,
        counts=counts,
        notes=notes,
    )

    if verbose:
        print(
            f"[{mod.key}] train {dataset.n_train} x {dataset.n_features} | "
            f"test {dataset.n_test} | classes {len(counts)}/{mod.n_classes}"
        )
        for note in notes:
            print(f"  note: {note}")

    return dataset
