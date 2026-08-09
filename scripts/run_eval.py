"""Run the baseline audio evaluator against an AutoAce batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import AnalysisResult
from app.services.audio_engine import analyze_audio_file


FIELDS = (
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
)
CATEGORICAL_FIELDS = (
    "emotional_tone",
    "emotional_intensity",
    "background_noise_severity",
    "audio_quality",
)
EMOTION_LABELS = ("neutral", "satisfied", "frustrated", "upset", "distressed")


def load_manifest(batch_dir: Path) -> list[dict[str, Any]]:
    """Load and validate the required name/result_json manifest."""
    manifest_path = batch_dir / "labels.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "result_json"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            filename = (row.get("name") or "").strip()
            if not filename:
                raise ValueError(f"Manifest line {line_number} has no name")

            raw_result = (row.get("result_json") or "").strip()
            expected = None
            if raw_result:
                try:
                    expected = json.loads(raw_result)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid result_json on manifest line {line_number}: {exc}"
                    ) from exc
                if not isinstance(expected, dict):
                    raise ValueError(
                        f"result_json on manifest line {line_number} must be an object"
                    )

            rows.append({"name": filename, "expected": expected})

    return rows


def validate_prediction(filename: str, raw_result: dict[str, Any]) -> dict[str, Any]:
    """Validate an engine result and return JSON-compatible data."""
    validated = AnalysisResult(**raw_result)
    result = validated.model_dump(mode="json")
    result["filename"] = filename
    return result


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate field-level accuracy and categorical classification metrics."""
    labeled = [record for record in records if record.get("expected") and record.get("prediction")]
    metrics: dict[str, Any] = {
        "labeled_files": len(labeled),
        "exact_schema_match_rate": 0.0,
        "fields": {},
    }
    if not labeled:
        return metrics

    exact_matches = 0
    for record in labeled:
        expected = record["expected"]
        prediction = record["prediction"]
        exact_matches += int(all(prediction.get(field) == expected.get(field) for field in FIELDS))

    metrics["exact_schema_match_rate"] = exact_matches / len(labeled)

    for field in FIELDS:
        y_true = [record["expected"].get(field) for record in labeled]
        y_pred = [record["prediction"].get(field) for record in labeled]
        metrics["fields"][field] = {
            "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        }

        if field in CATEGORICAL_FIELDS:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=None,
                labels=sorted(set(y_true) | set(y_pred)),
                zero_division=0,
            )
            labels = sorted(set(y_true) | set(y_pred))
            metrics["fields"][field].update(
                {
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                    "per_class": {
                        label: {
                            "precision": float(precision[index]),
                            "recall": float(recall[index]),
                            "f1": float(f1[index]),
                            "support": int(support[index]),
                        }
                        for index, label in enumerate(labels)
                    },
                }
            )

    return metrics


def write_confusion_matrix(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write the emotional-tone confusion matrix as CSV and PNG."""
    labeled = [record for record in records if record.get("expected") and record.get("prediction")]
    y_true = [record["expected"]["emotional_tone"] for record in labeled]
    y_pred = [record["prediction"]["emotional_tone"] for record in labeled]
    matrix = confusion_matrix(y_true, y_pred, labels=list(EMOTION_LABELS))

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected\\predicted", *EMOTION_LABELS])
        writer.writerows([label, *row.tolist()] for label, row in zip(EMOTION_LABELS, matrix))

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.figure.colorbar(image, ax=axis)
        axis.set(
            xticks=np.arange(len(EMOTION_LABELS)),
            yticks=np.arange(len(EMOTION_LABELS)),
            xticklabels=EMOTION_LABELS,
            yticklabels=EMOTION_LABELS,
            ylabel="Expected",
            xlabel="Predicted",
            title="Emotional Tone Confusion Matrix",
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    except Exception as exc:  # The CSV remains available if plotting is unavailable.
        output_path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def run_batch(batch_dir: Path, output_dir: Path, use_ser: bool = False) -> dict[str, Any]:
    """Process every manifest row and write baseline artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(batch_dir)
    records: list[dict[str, Any]] = []
    total_audio_seconds = 0.0
    started = time.perf_counter()

    for row in rows:
        filename = row["name"]
        file_path = batch_dir / filename
        record: dict[str, Any] = {"filename": filename, "expected": row["expected"]}
        if not file_path.is_file():
            record["error"] = "File listed in manifest but missing from batch."
            records.append(record)
            continue

        try:
            total_audio_seconds += float(sf.info(file_path).duration)
            raw_prediction = analyze_audio_file(str(file_path), use_ser=use_ser)
            if raw_prediction is None:
                raise RuntimeError("Audio engine returned no result")
            record["prediction"] = validate_prediction(filename, raw_prediction)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    elapsed_seconds = time.perf_counter() - started
    predictions = [record["prediction"] for record in records if record.get("prediction")]
    errors = [record for record in records if record.get("error")]
    metrics = calculate_metrics(records)
    metrics.update(
        {
            "engine": "hybrid_ser" if use_ser else "baseline_heuristic",
            "processed_files": len(predictions),
            "error_count": len(errors),
            "total_audio_seconds": total_audio_seconds,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_audio_minute": (
                elapsed_seconds / (total_audio_seconds / 60) if total_audio_seconds else None
            ),
            "prediction_counts": dict(Counter(prediction["emotional_tone"] for prediction in predictions)),
        }
    )

    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        if predictions:
            writer = csv.DictWriter(handle, fieldnames=["filename", *FIELDS, "confidence"])
            writer.writeheader()
            writer.writerows(predictions)
    (output_dir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_matrix(records, output_dir / "emotional_tone_confusion_matrix.png")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="Directory containing audio files and labels.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "baseline",
        help="Directory for predictions and metrics",
    )
    parser.add_argument(
        "--engine",
        choices=("baseline", "hybrid"),
        default="baseline",
        help="Inference engine: baseline heuristics or local pretrained SER hybrid",
    )
    args = parser.parse_args()
    metrics = run_batch(
        args.batch_dir.resolve(),
        args.output_dir.resolve(),
        use_ser=args.engine == "hybrid",
    )
    print(json.dumps(metrics, indent=2))
    return 1 if metrics["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
