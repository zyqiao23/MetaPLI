"""Evaluate complete prediction tables without filtering or recalibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import target_metrics


def _json_values(values: dict) -> dict:
    # JSON null represents an undefined correlation; CSV retains NaN.
    return {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in values.items()
    }


def run(
    predictions_path: Path,
    *,
    metadata_path: Path,
    split: str = "test",
    per_target_output: Path | None = None,
    summary_output: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if metadata_path is None:
        raise ValueError("Metadata is required to verify the complete split")
    predictions = pd.read_csv(predictions_path, dtype={"query_uid": str, "target_id": str})
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path, dtype={"uid": str, "Uniprot": str, "Split": str})
        required = {"uid", "Uniprot", "Split", "-logAffi"}
        if not required.issubset(metadata.columns):
            raise ValueError(f"Metadata requires {sorted(required)}")
        if metadata["uid"].isna().any() or metadata["uid"].duplicated().any():
            raise ValueError("Metadata query identifiers must be present and unique")
        expected = metadata.loc[metadata["Split"].eq(split)].set_index("uid")
        if expected.empty:
            raise ValueError(f"No metadata records in split {split!r}")
        if "query_uid" not in predictions or predictions["query_uid"].duplicated().any():
            raise ValueError("Prediction query identifiers must be present and unique")
        actual_ids = set(predictions["query_uid"])
        expected_ids = set(expected.index)
        if actual_ids != expected_ids:
            raise ValueError(
                "Predictions do not contain the complete requested split: "
                f"missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)}"
            )
        truth = pd.to_numeric(expected.loc[predictions["query_uid"], "-logAffi"], errors="raise").to_numpy()
        target = expected.loc[predictions["query_uid"], "Uniprot"].to_numpy()
        if "target_id" in predictions and not np.array_equal(predictions["target_id"].to_numpy(), target):
            raise ValueError("Prediction targets disagree with metadata")
        if "targets" in predictions:
            stored_truth = pd.to_numeric(predictions["targets"], errors="raise").to_numpy()
            # Missing truth may be supplied later by metadata. Existing truth
            # must agree; a prediction is never used to modify a measurement.
            present = ~np.isnan(stored_truth)
            if not np.isfinite(stored_truth[present]).all() or not np.array_equal(stored_truth[present], truth[present]):
                raise ValueError("Prediction labels disagree with metadata")
        predictions["target_id"] = target
        predictions["targets"] = truth
    per_target, summary = target_metrics(predictions)
    if per_target_output is not None:
        per_target_output.parent.mkdir(parents=True, exist_ok=True)
        per_target.to_csv(per_target_output, index=False)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(_json_values(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return per_target, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-target-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    per_target, summary = run(
        args.predictions, metadata_path=args.metadata, split=args.split,
        per_target_output=args.per_target_output, summary_output=args.summary_output,
    )
    print(per_target.to_csv(index=False), end="")
    print(json.dumps(_json_values(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
