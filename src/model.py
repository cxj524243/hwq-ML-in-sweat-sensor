"""Classifier construction, training and metric computation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import config
from src.dataset import Dataset


@dataclass
class TrainResult:
    model: object
    scaler: StandardScaler
    encoder: LabelEncoder
    y_pred: np.ndarray
    y_pred_values: np.ndarray
    y_train_encoded: np.ndarray
    y_test_encoded: np.ndarray
    X_train_scaled: np.ndarray
    accuracy: float
    macro_f1: float
    weighted_f1: float
    mae: float
    confusion: np.ndarray
    report: dict = field(default_factory=dict)

    def summary(self, mod: config.Modality) -> dict:
        return {
            "modality": mod.key,
            "label": mod.label,
            "n_classes": mod.n_classes,
            "n_train": int(self.X_train_scaled.shape[0]),
            "n_test": int(len(self.y_test_encoded)),
            "n_features": int(self.X_train_scaled.shape[1]),
            "classifier": mod.classifier,
            "accuracy": round(float(self.accuracy), 6),
            "macro_f1": round(float(self.macro_f1), 6),
            "weighted_f1": round(float(self.weighted_f1), 6),
            "mae": round(float(self.mae), 6),
        }


def build_classifier(mod: config.Modality):
    if mod.classifier == "svc":
        return SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            random_state=config.SEED,
            probability=True,
        )
    if mod.classifier == "knn":
        return KNeighborsClassifier(
            n_neighbors=config.KNN_K, weights=config.KNN_WEIGHTS
        )
    raise ValueError(f"unknown classifier {mod.classifier!r}")


def train_and_evaluate(dataset: Dataset, mod: config.Modality) -> TrainResult:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(dataset.X_train)
    X_test_scaled = scaler.transform(dataset.X_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(dataset.y_train)
    y_test_encoded = encoder.transform(dataset.y_test)

    model = build_classifier(mod)
    model.fit(X_train_scaled, y_train_encoded)
    y_pred = model.predict(X_test_scaled)

    y_pred_values = encoder.inverse_transform(y_pred)
    target_names = [str(c) for c in encoder.classes_]

    accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(dataset.y_test, y_pred_values)
    cm = confusion_matrix(y_test_encoded, y_pred, labels=list(range(len(encoder.classes_))))
    report = classification_report(
        y_test_encoded,
        y_pred,
        labels=list(range(len(encoder.classes_))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return TrainResult(
        model=model,
        scaler=scaler,
        encoder=encoder,
        y_pred=y_pred,
        y_pred_values=y_pred_values,
        y_train_encoded=y_train_encoded,
        y_test_encoded=y_test_encoded,
        X_train_scaled=X_train_scaled,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        mae=float(mae),
        confusion=cm,
        report=report,
    )
