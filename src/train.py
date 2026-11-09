"""Train and evaluate one analyte.

Usage (from the project root):

    python -m src.train --modality glu
    python -m src.train --modality mg --no-abnormal
    python src/train.py --modality ph --data-root D:/some/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402

import config  # noqa: E402
from src import plots  # noqa: E402
from src.dataset import build_dataset  # noqa: E402
from src.io_utils import ensure_dirs  # noqa: E402
from src.model import train_and_evaluate  # noqa: E402


def run_modality(
    mod: config.Modality,
    data_root: Path | None = None,
    inject_abnormal: bool | None = None,
    save_model: bool = True,
    make_figures: bool = True,
    verbose: bool = True,
) -> dict:
    ensure_dirs(config.RESULT_DIR, config.FIGURE_DIR, config.MODEL_DIR)

    dataset = build_dataset(mod, data_root=data_root, inject_abnormal=inject_abnormal)
    result = train_and_evaluate(dataset, mod)
    summary = result.summary(mod)

    if verbose:
        print("-" * 62)
        print(f"{mod.label} ({mod.key}) | {mod.classifier.upper()} "
              f"| classes={mod.n_classes} | features={summary['n_features']}")
        print(f"train={summary['n_train']}  test={summary['n_test']}")
        print(f"accuracy   : {result.accuracy:.4f} ({result.accuracy * 100:.2f}%)")
        print(f"macro F1   : {result.macro_f1:.4f}")
        print(f"weighted F1: {result.weighted_f1:.4f}")
        print(f"ordinal MAE: {result.mae:.4f}")
        print("-" * 62)

    summary["per_class"] = {
        str(k): {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"]),
        }
        for k, v in result.report.items()
        if k not in {"accuracy", "macro avg", "weighted avg"}
    }
    summary["confusion"] = result.confusion.tolist()
    summary["class_order"] = [float(c) for c in result.encoder.classes_]
    summary["notes"] = dataset.notes

    out_json = config.RESULT_DIR / f"{mod.key}_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if make_figures:
        plots.plot_confusion(
            result, f"{mod.label}", config.FIGURE_DIR / f"confusion_{mod.key}.png"
        )
        plots.plot_per_class_accuracy(
            result, f"{mod.label}", config.FIGURE_DIR / f"per_class_acc_{mod.key}.png"
        )
        plots.plot_pca(
            result, f"{mod.label}", config.FIGURE_DIR / f"pca_{mod.key}.png"
        )
        plots.plot_pred_vs_true(
            result, f"{mod.label}", mod.unit, config.FIGURE_DIR / f"scatter_{mod.key}.png"
        )

    if save_model:
        joblib.dump(result.model, config.MODEL_DIR / f"{mod.key}_{mod.classifier}_model.pkl")
        joblib.dump(result.scaler, config.MODEL_DIR / f"{mod.key}_scaler.pkl")
        joblib.dump(result.encoder, config.MODEL_DIR / f"{mod.key}_label_encoder.pkl")

    if verbose:
        print(f"[saved] {out_json}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one analyte classifier.")
    parser.add_argument(
        "--modality",
        required=True,
        choices=sorted(config.MODALITIES) + ["glucose", "magnesium", "chloride", "urea"],
        help="which analyte to run",
    )
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override the image root (default: ./data)")
    parser.add_argument("--no-abnormal", action="store_true",
                        help="disable the abnormal-ROI stress injection")
    parser.add_argument("--no-model", action="store_true", help="do not save .pkl files")
    parser.add_argument("--no-figures", action="store_true", help="do not render figures")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    mod = config.get_modality(args.modality)
    inject = False if args.no_abnormal else None
    run_modality(
        mod,
        data_root=args.data_root,
        inject_abnormal=inject,
        save_model=not args.no_model,
        make_figures=not args.no_figures,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
