"""Counting error against ground-truth counts.

Ground truth is a CSV file with the columns ``file`` (image or video file name, no
directory) and ``count`` (the true number of fruits in it).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fruit_counter.exceptions import InputError
from fruit_counter.sources import InputSource, SourceKind, list_images


def load_ground_truth(path: str | Path) -> dict[str, int]:
    path = Path(path)
    truth: dict[str, int] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"file", "count"} <= set(reader.fieldnames or ()):
                raise InputError(f"{path} must have a header row with 'file' and 'count' columns")
            for line, row in enumerate(reader, start=2):
                name = (row["file"] or "").strip()
                count = (row["count"] or "").strip()
                if not name or not count.isdigit():
                    raise InputError(
                        f"{path}, line {line}: expected a file name and a non-negative integer"
                    )
                truth[name] = int(count)
    except OSError as exc:
        raise InputError(f"Cannot read ground truth {path}: {exc}") from exc
    if not truth:
        raise InputError(f"{path} contains no counts")
    return truth


def check_coverage(source: InputSource, truth: Mapping[str, int]) -> None:
    """Fail before processing if none of the source's files have a ground-truth count."""
    if source.path is None:
        raise InputError("Ground truth needs file inputs; it cannot be used with a camera")
    names = (
        [p.name for p in list_images(source.path)]
        if source.kind is SourceKind.IMAGE_DIR
        else [source.path.name]
    )
    if not any(name in truth for name in names):
        raise InputError(f"None of the input files are listed in the ground truth: {source}")


def evaluate_counts(predicted: Mapping[str, int], truth: Mapping[str, int]) -> dict[str, Any]:
    """Compare predicted and true counts for the files present in both."""
    pairs = [(name, truth[name], count) for name, count in predicted.items() if name in truth]
    if not pairs:
        raise InputError("None of the processed files are listed in the ground truth")
    errors = [count - true for _, true, count in pairs]
    return {
        "files_evaluated": len(pairs),
        "files_without_ground_truth": sorted(set(predicted) - set(truth)),
        "mean_absolute_error": round(sum(abs(e) for e in errors) / len(errors), 3),
        # Negative means undercounting.
        "mean_error": round(sum(errors) / len(errors), 3),
        "total_true": sum(true for _, true, _ in pairs),
        "total_predicted": sum(count for _, _, count in pairs),
        "items": [
            {"file": name, "true_count": true, "predicted_count": count, "error": count - true}
            for name, true, count in pairs
        ],
    }
