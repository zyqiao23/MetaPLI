"""Predict every query in a configured split using a frozen MetaPLI model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import yaml

from .context_policies import PolicyContextIndex
from .data import LMDBGraphStore
from .model import MetaPLI


def _validate_keys(config: dict, allowed: set[str], required: set[str], section: str) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{section} must be a mapping")
    unknown = sorted(set(config) - allowed)
    missing = sorted(required - set(config))
    if unknown or missing:
        raise ValueError(f"Invalid {section}: unknown={unknown}, missing={missing}")


def _context_config(config: dict) -> dict:
    _validate_keys(config, {"k", "policy", "seed"}, {"k", "policy", "seed"}, "context")
    if config["policy"] not in ("FPSimTopK", "FPSimNorm", "Random", "ActivityBased"):
        raise ValueError("Unknown context policy")
    for key in ("k", "seed"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"context.{key} must be an integer")
    if config["k"] < 1:
        raise ValueError("context.k must be positive")
    return dict(config)


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def _resolve(config_path: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("File paths must be nonempty strings")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path) -> tuple[Path, Path]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    allowed = {"checkpoint", "metadata_csv", "similarity_csv", "graph_lmdb", "output_csv", "split", "device", "seed", "context"}
    required = {"checkpoint", "metadata_csv", "graph_lmdb", "output_csv", "split", "device", "seed", "context"}
    _validate_keys(config, allowed, required, "prediction config")
    context_config = _context_config(config["context"])
    _seed(config["seed"])
    paths = {key: _resolve(config_path, config[key]) for key in ("checkpoint", "metadata_csv", "graph_lmdb", "output_csv")}
    similarity_path = _resolve(config_path, config["similarity_csv"]) if config.get("similarity_csv") is not None else None
    for path in (paths["checkpoint"], paths["metadata_csv"], paths["graph_lmdb"]):
        if not path.exists():
            raise FileNotFoundError(path)
    output_path = paths["output_csv"]
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")
    split = config["split"]
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a nonempty string")
    index = PolicyContextIndex.from_csv(paths["metadata_csv"], similarity_path, **context_config)
    query_uids = index.query_uids(split=split)
    if not query_uids:
        raise ValueError(f"No query records in split {split!r}")
    device = torch.device(config["device"])
    model = MetaPLI.from_checkpoint(paths["checkpoint"], map_location=device, context_num=context_config["k"]).to(device).eval()
    rows = []
    with LMDBGraphStore(paths["graph_lmdb"]) as store, torch.no_grad():
        for query_uid in query_uids:
            episode = index.episode(query_uid)
            output = model(
                dgl.batch([store.get(query_uid)]).to(device),
                dgl.batch([store.get(uid) for uid in episode.context_uids]).to(device),
                None,
                torch.tensor([episode.context_labels], dtype=torch.float32, device=device),
                torch.tensor([0], dtype=torch.long, device=device),
            )
            mu, sigma, mask = output[1], output[2], output[8]
            if mask.any() or mu.numel() != 1 or sigma.numel() != 1:
                raise ValueError(f"Unexpected prediction shape for {query_uid}")
            if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
                raise ValueError(f"Invalid predictive distribution for {query_uid}")
            truth = float(episode.query_label)
            if np.isinf(truth):
                raise ValueError(f"Infinite measured activity for {query_uid}")
            target = str(index.rows.loc[query_uid, index.target_col])
            standard_deviation = float(sigma.item())
            rows.append({
                "query_uid": query_uid, "target_id": target, "pdb_id": episode.group_id,
                "preds": float(mu.item()), "targets": truth,
                "sigma": standard_deviation, "variance": standard_deviation**2,
                "context_uids": "#".join(episode.context_uids),
                "context_k": context_config["k"], "context_policy": context_config["policy"],
                "context_seed": context_config["seed"], "seed": config["seed"],
            })
    if len(rows) != len(query_uids) or {row["query_uid"] for row in rows} != set(query_uids):
        raise AssertionError("Predictions must contain the entire requested split")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    manifest = {
        "config": config, "n_queries": len(query_uids), "n_predictions": len(rows),
        "checkpoint_sha256": _file_sha256(paths["checkpoint"]),
        "metadata_sha256": _file_sha256(paths["metadata_csv"]),
        "similarity_sha256": _file_sha256(similarity_path) if similarity_path is not None else None,
        "predictions_sha256": _file_sha256(output_path),
        "torch": torch.__version__, "dgl": dgl.__version__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output, manifest = run(args.config)
    print(f"predictions={output}\nmanifest={manifest}")


if __name__ == "__main__":
    main()
