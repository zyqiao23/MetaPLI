"""Combine prepared records with the archived Exact-target or Homology split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


RECORD_COLUMNS = ("uid", "Uniprot", "pdb_id", "-logAffi", "Molecule SMILES", "pocket_path", "ligand_path")


def prepare_metadata(records_path: str | Path, split_path: str | Path,
                     output_path: str | Path) -> pd.DataFrame:
    """Join one split assignment for every record; no record is discarded.

    Input activity is already normalized pIC50 (or missing for prediction).
    Exact-target/Homology partition assignments come from their respective
    released manifests. The split file has exactly two columns: uid and Split.
    """
    records_path, split_path, output_path = map(Path, (records_path, split_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    records = pd.read_csv(records_path, dtype={
        "uid": str, "Uniprot": str, "pdb_id": str,
        "Molecule SMILES": str, "pocket_path": str, "ligand_path": str,
    })
    splits = pd.read_csv(split_path, dtype={"uid": str, "Split": str})
    missing = sorted(set(RECORD_COLUMNS) - set(records.columns))
    if missing:
        raise ValueError(f"Records are missing columns: {missing}")
    if set(splits.columns) != {"uid", "Split"}:
        raise ValueError("Split manifest must have exactly uid and Split columns")
    if "Split" in records.columns:
        raise ValueError("Records already contain Split; supply the original prepared records")
    if records.empty:
        raise ValueError("Records are empty")
    for frame in (records, splits):
        if frame.uid.isna().any() or frame.uid.str.strip().eq("").any() or frame.uid.duplicated().any():
            raise ValueError("Record and split UIDs must be nonempty and unique")
    for column in set(RECORD_COLUMNS) - {"-logAffi"}:
        if (records[column].isna() | records[column].astype(str).str.strip().eq("")).any():
            raise ValueError(f"Missing values in {column!r}")
    activity = pd.to_numeric(records["-logAffi"], errors="raise")
    if np.isinf(activity.to_numpy(dtype=float)).any():
        raise ValueError("Activities must be finite or missing")
    records["-logAffi"] = activity
    if not splits.Split.isin(["train", "validation", "test"]).all():
        raise ValueError("Split must be train, validation or test")
    record_uids, split_uids = set(records.uid), set(splits.uid)
    if record_uids != split_uids:
        raise ValueError("Split manifest must contain exactly the same UIDs as the records")
    merged = records.merge(splits, on="uid", how="left", validate="one_to_one").sort_values("uid")
    if merged.groupby("Uniprot").Split.nunique().gt(1).any():
        raise ValueError("A target cannot span multiple train/validation/test partitions")
    # Preserve all records and their graph paths when relocating the metadata.
    for column in ("pocket_path", "ligand_path"):
        merged[column] = merged[column].map(
            lambda value: os.path.relpath(
                (records_path.parent / value).resolve() if not Path(value).is_absolute()
                else Path(value).resolve(), output_path.parent.resolve(),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_metadata(args.records, args.split, args.output)
    print(f"Prepared {len(output)} records: {args.output}")


if __name__ == "__main__":
    main()
