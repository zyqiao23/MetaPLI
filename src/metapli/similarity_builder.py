"""Build all eligible ECFP4/Tanimoto pairs using the manuscript fingerprint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def build_similarity(metadata_path: str | Path, output_path: str | Path) -> dict:
    """Write every unordered pair sharing a target, structure source and split.

    Fingerprints use radius 2, 1024 bits and no chirality. Invalid SMILES fail
    the whole operation; they cannot remove queries or their contexts.
    """
    metadata_path, output_path = Path(metadata_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    metadata = pd.read_csv(metadata_path, dtype={
        "uid": str, "Uniprot": str, "pdb_id": str, "Split": str,
        "Molecule SMILES": str,
    })
    required = {"uid", "Uniprot", "pdb_id", "Split", "Molecule SMILES"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    if metadata.empty or metadata.uid.duplicated().any():
        raise ValueError("Metadata must contain nonempty, unique UIDs")
    for column in required:
        if (metadata[column].isna() | metadata[column].str.strip().eq("")).any():
            raise ValueError(f"Metadata contains missing values in {column!r}")
    fingerprints = {}
    failures = []
    for row in metadata.sort_values("uid").to_dict("records"):
        molecule = Chem.MolFromSmiles(row["Molecule SMILES"])
        if molecule is None:
            failures.append(row["uid"])
        else:
            fingerprints[row["uid"]] = AllChem.GetMorganFingerprintAsBitVect(
                molecule, radius=2, nBits=1024, useChirality=False,
            )
    if failures:
        raise ValueError(f"Could not parse SMILES for UIDs: {failures}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    written = 0
    groups = metadata.groupby(["Uniprot", "pdb_id", "Split"], sort=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pdb_id", "uid1", "uid2", "TanimotoSimilarity"])
            for (_, source, _), rows in groups:
                uids = sorted(rows.uid.tolist())
                for index, uid in enumerate(uids[:-1]):
                    others = uids[index + 1:]
                    scores = DataStructs.BulkTanimotoSimilarity(
                        fingerprints[uid], [fingerprints[other] for other in others],
                    )
                    writer.writerows((source, uid, other, f"{float(score):.12g}")
                                     for other, score in zip(others, scores))
                    written += len(others)
        Path(temporary).rename(output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"rows": len(metadata), "groups": len(groups), "pairs": written,
            "fingerprint": "ECFP4", "radius": 2, "bits": 1024,
            "similarity": "Tanimoto"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_similarity(args.metadata, args.output), indent=2))


if __name__ == "__main__":
    main()
