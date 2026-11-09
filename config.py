"""Central configuration for the multi-analyte sweat-sensor classification project.

Every value in this file is taken verbatim from the original per-modality
training scripts that produced the reported results:

    glu  <- legacy/glu_ts1.py
    Mg   <- legacy/Mg_tr11.py
    Cl   <- legacy/cl_tr3.py
    pH   <- legacy/pH_tr5.py
    urea <- legacy/ur_tr4tu.py

Keep this file as the single source of truth: the training code reads only
from here, so changing a number here is the only way to change an experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data root can be overridden for a checkout that keeps images elsewhere,
# e.g.  set SWEAT_DATA_ROOT=D:\some\other\images
def _resolve_data_root() -> Path:
    """Pick the image root: env override > ./data > ../data.

    Keeping the images one level above the project (as on the author's machine)
    lets the code folder stay small enough to version-control.
    """
    env = os.environ.get("SWEAT_DATA_ROOT")
    if env:
        return Path(env)
    local = BASE_DIR / "data"
    if local.is_dir():
        return local
    sibling = BASE_DIR.parent / "data"
    if sibling.is_dir():
        return sibling
    return local


DATA_ROOT = _resolve_data_root()
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"

# ---------------------------------------------------------------------------
# Shared experimental protocol
# ---------------------------------------------------------------------------
GRID_SIZE = 3           # ROI is split into 3x3 cells -> 27 RGB-mean features
IMAGES_PER_CONC = 10    # 10 photos per concentration level
TRAIN_PER_CONC = 7      # first 7 (by shooting index) build the training set
TEST_PER_CONC = 3       # last 3 build the test set  -> 7:3 split, no val set
SEED = 42

SVM_C = 10.0
SVM_GAMMA = "scale"
SVM_KERNEL = "rbf"

KNN_K = 3
KNN_WEIGHTS = "distance"

NORMALISE_TO_UNIT = True   # features are divided by 255 before scaling


@dataclass(frozen=True)
class Modality:
    key: str                  # short id used on the command line
    label: str                # human readable name (Chinese, for figures)
    unit: str                 # concentration unit shown in plots
    data_subdir: str          # folder under DATA_ROOT
    file_prefix: str          # files are named "<prefix><value>_<index>.jpg"
    values: tuple[float, ...]  # concentration levels == classes, in order
    roi: tuple[int, int, int, int]   # (x_start, y_start, x_end, y_end) in pixels
    train_sigmas: tuple[float, ...]
    train_reps: int
    test_sigmas: tuple[float, ...]
    test_reps: int
    classifier: str = "svc"          # "svc" | "knn"
    inject_abnormal: bool = False    # overwrite random test rows with noise ROI
    abnormal_count: int = 0          # how many test rows are overwritten
    blank_at: float | None = None    # prepend one white ROI to this class's test set

    @property
    def n_classes(self) -> int:
        return len(self.values)

    @property
    def per_class_train(self) -> int:
        return (1 + len(self.train_sigmas) * self.train_reps) * TRAIN_PER_CONC

    @property
    def per_class_test(self) -> int:
        return (1 + len(self.test_sigmas) * self.test_reps) * TEST_PER_CONC


MODALITIES: dict[str, Modality] = {
    "glu": Modality(
        key="glu",
        label="Glucose",
        unit="mM",
        data_subdir="glu",
        file_prefix="glu=",
        values=(0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0),
        roi=(1292, 1817, 1647, 2156),
        train_sigmas=(0.5, 1.0, 1.5, 2.0, 2.5),
        train_reps=4,
        test_sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
        test_reps=2,
        classifier="svc",
        inject_abnormal=True,
        abnormal_count=2,
    ),
    "mg": Modality(
        key="mg",
        label="Magnesium",
        unit="mM",
        data_subdir="Mg",
        file_prefix="mg=",
        # NB: images also exist for 10 mM but the original protocol stops at 8 mM.
        values=(0, 0.5, 1, 2, 4, 5, 8),
        roi=(1528, 1804, 1542, 1815),
        train_sigmas=(1.0, 1.5, 2.0, 2.5, 3.0),
        train_reps=3,
        test_sigmas=(1.0, 1.5, 2.0, 2.5, 3.0),
        test_reps=3,
        classifier="knn",
        inject_abnormal=False,
        blank_at=5,
    ),
    "cl": Modality(
        key="cl",
        label="Chloride",
        unit="mM",
        data_subdir="Cl",
        file_prefix="cl=",
        values=(0, 5, 15, 30, 45, 60, 75, 90, 100),
        roi=(1441, 2039, 1604, 2209),
        train_sigmas=(0.5, 1.0, 1.5, 2.0, 2.5),
        train_reps=4,
        test_sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
        test_reps=2,
        classifier="svc",
        inject_abnormal=True,
        abnormal_count=2,
    ),
    "ph": Modality(
        key="ph",
        label="pH",
        unit="",
        data_subdir="pH",
        file_prefix="pH=",
        values=(3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8),
        roi=(1441, 2039, 1604, 2209),
        train_sigmas=(0.5, 1.0, 1.5, 2.0, 2.5),
        train_reps=4,
        test_sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
        test_reps=2,
        classifier="svc",
        inject_abnormal=True,
        abnormal_count=1,
    ),
    "ur": Modality(
        key="ur",
        label="Urea",
        unit="mM",
        data_subdir="urea",
        file_prefix="ur=",
        values=(0, 5, 10, 15, 20, 25, 30, 50, 80, 100),
        roi=(1329, 1696, 1673, 2028),
        train_sigmas=(2.0, 3.0, 4.0, 5.0, 6.0),
        train_reps=4,
        test_sigmas=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        test_reps=2,
        classifier="svc",
        inject_abnormal=True,
        abnormal_count=4,
    ),
}

ALIASES = {
    "glucose": "glu",
    "mg": "mg",
    "magnesium": "mg",
    "cl": "cl",
    "chloride": "cl",
    "ph": "ph",
    "ur": "ur",
    "urea": "ur",
}


def get_modality(name: str) -> Modality:
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in MODALITIES:
        raise KeyError(
            f"unknown modality {name!r}; choose from {sorted(MODALITIES)}"
        )
    return MODALITIES[key]
