"""Run all five analytes and write the aggregate result table.

    python run_all.py                # full pipeline, figures + models + csv
    python run_all.py --no-figures   # faster, numbers only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.train import run_modality  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train all five analytes.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--no-abnormal", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for key in config.MODALITIES:
        mod = config.MODALITIES[key]
        summary = run_modality(
            mod,
            data_root=args.data_root,
            inject_abnormal=False if args.no_abnormal else None,
            save_model=not args.no_model,
            make_figures=not args.no_figures,
        )
        rows.append(summary)
        print()

    csv_path = config.RESULT_DIR / "metrics.csv"
    fields = [
        "modality", "label", "classifier", "n_classes", "n_features",
        "n_train", "n_test", "accuracy", "macro_f1", "weighted_f1", "mae",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("=" * 72)
    print(f"{'analyte':<10}{'clf':<6}{'classes':>8}{'train':>8}{'test':>7}"
          f"{'acc':>9}{'macroF1':>9}{'MAE':>8}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['modality']:<10}{row['classifier']:<6}{row['n_classes']:>8}"
            f"{row['n_train']:>8}{row['n_test']:>7}"
            f"{row['accuracy'] * 100:>8.2f}%{row['macro_f1']:>9.3f}{row['mae']:>8.3f}"
        )
    print("=" * 72)
    print(f"[saved] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
