"""Build model input graphs from prepared pocket MOL2 and ligand SDF files.

Each supplied metadata row is processed. A failed structure raises after writing
its error in the row manifest; the graph store is published only if every row
was built successfully. Pose generation and pocket extraction are external.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import dgl
import lmdb
import numpy as np
import pandas as pd
import rdkit
import torch
from rdkit import Chem


ATOM_TYPES = {symbol: index for index, symbol in enumerate(
    ["C", "N", "O", "F", "P", "S", "Cl", "Br"]
)}
HYBRIDIZATIONS = {name: index for index, name in enumerate(
    ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
)}
CHIRALITIES = {name: index for index, name in enumerate(
    ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
)}
BOND_TYPES = {name: index for index, name in enumerate(
    ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
)}
STEREO_TYPES = {name: index for index, name in enumerate(
    ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]
)}

# Exact period/group coverage in the feature table.
PERIOD_GROUP = {
    1: (1, 1), 2: (1, 18),
    3: (2, 1), 4: (2, 2), 5: (2, 13), 6: (2, 14), 7: (2, 15),
    8: (2, 16), 9: (2, 17), 10: (2, 18),
    11: (3, 1), 12: (3, 2), 13: (3, 13), 14: (3, 14), 15: (3, 15),
    16: (3, 16), 17: (3, 17), 18: (3, 18),
    19: (4, 1), 20: (4, 2), 21: (4, 3), 22: (4, 4), 23: (4, 5),
    24: (4, 6), 25: (4, 7), 26: (4, 8), 27: (4, 9), 28: (4, 10),
    29: (4, 11), 30: (4, 12), 31: (4, 13), 32: (4, 14),
    33: (4, 15), 34: (4, 16), 35: (4, 17), 36: (4, 18),
    37: (5, 1), 38: (5, 2), 39: (5, 3), 40: (5, 4), 53: (5, 17),
}


def _one_hot(index: int | None, size: int) -> list[int]:
    return [int(position == index) for position in range(size)]


def atom_features(atom: Chem.Atom) -> list[float]:
    atomic_number = atom.GetAtomicNum()
    if atomic_number not in PERIOD_GROUP:
        raise ValueError(
            f"Atomic number {atomic_number} is absent from the feature table"
        )
    period, group = PERIOD_GROUP[atomic_number]
    categorical = (
        (ATOM_TYPES.get(atom.GetSymbol(), 8), 9),
        (int(atom.GetDegree()), 6),
        (int(atom.GetFormalCharge()) + 2, 5),
        (CHIRALITIES.get(str(atom.GetChiralTag()), 3), 4),
        (int(atom.GetTotalNumHs()), 5),
        (HYBRIDIZATIONS.get(str(atom.GetHybridization()), 4), 5),
        (period - 1, 6),
        (group - 1, 18),
    )
    values: list[float] = []
    for index, size in categorical:
        values.extend(_one_hot(index, size))
    periodic_table = Chem.GetPeriodicTable()
    values.extend(
        [
            atom.GetMass() / 100.0,
            atom.GetExplicitValence() / 10.0,
            atom.GetImplicitValence() / 10.0,
            periodic_table.GetRvdw(atomic_number),
            float(atom.GetIsAromatic()),
        ]
    )
    hbond = atom.GetProp("hbonds").split() if atom.HasProp("hbonds") else []
    values.extend([float("HA" in hbond), float("HD" in hbond)])
    if len(values) != 65:
        raise AssertionError(f"Expected 65 atom features, found {len(values)}")
    return values


def edge_features(bond: Chem.Bond | None = None) -> list[float]:
    if bond is None:
        return [0.0] * 13
    values: list[float] = []
    values.extend(_one_hot(BOND_TYPES.get(str(bond.GetBondType())), 4))
    values.extend(_one_hot(STEREO_TYPES.get(str(bond.GetStereo())), 6))
    values.extend(
        [float(bond.GetIsConjugated()), float(bond.IsInRing()), 1.0]
    )
    if len(values) != 13:
        raise AssertionError(f"Expected 13 edge features, found {len(values)}")
    return values


def annotate_hbonds(ligand, pocket) -> None:
    """Apply the ProLIF donor/acceptor annotation in place."""
    import prolif as plf  # optional graph-building dependency

    interactions = plf.Fingerprint(["HBDonor", "HBAcceptor"]).generate(
        ligand, pocket, metadata=True
    )
    ligand_types = {"HD": set(), "HA": set()}
    pocket_types = {"HD": set(), "HA": set()}
    for residue_interactions in interactions.values():
        for interaction_type, instances in residue_interactions.items():
            if not instances:
                continue
            # Preserve the inspected graph representation: it consumes
            # the first metadata record for each interaction type/residue pair.
            indices = instances[0]["parent_indices"]
            if interaction_type == "HBDonor":
                ligand_types["HD"].update(indices["ligand"])
                pocket_types["HA"].update(indices["protein"])
            elif interaction_type == "HBAcceptor":
                ligand_types["HA"].update(indices["ligand"])
                pocket_types["HD"].update(indices["protein"])
    for molecule, assignments in ((ligand, ligand_types), (pocket, pocket_types)):
        for label, atom_indices in assignments.items():
            for index in atom_indices:
                atom = molecule.GetAtomWithIdx(int(index))
                if atom.GetAtomicNum() == 1:
                    continue
                existing = atom.GetProp("hbonds") if atom.HasProp("hbonds") else ""
                atom.SetProp("hbonds", f"{existing} {label}")


def _read_molecules(pocket_path: Path, ligand_path: Path):
    from prolif.molecule import Molecule

    supplier = Chem.SDMolSupplier(
        str(ligand_path), sanitize=False, removeHs=False
    )
    ligand_rdkit = supplier[0] if len(supplier) else None
    if ligand_rdkit is None:
        raise ValueError(f"Could not parse ligand SDF: {ligand_path}")
    for atom in ligand_rdkit.GetAtoms():
        atom.UpdatePropertyCache(strict=False)

    pocket_block = pocket_path.read_text(encoding="utf-8")
    pocket_rdkit = Chem.MolFromMol2Block(
        pocket_block, sanitize=False, removeHs=False
    )
    if pocket_rdkit is None:
        # Some protein MOL2 files contain valid coordinates and bonds but fail
        # RDKit's optional functional-group cleanup (for example, an O.co2
        # atom with an unusual local valence).  The builder already
        # disabled sanitization, so retrying without cleanup is the narrowest
        # parser fallback that preserves every atom, coordinate, and bond.
        pocket_rdkit = Chem.MolFromMol2Block(
            pocket_block,
            sanitize=False,
            removeHs=False,
            cleanupSubstructures=False,
        )
    if pocket_rdkit is None:
        raise ValueError(f"Could not parse pocket MOL2: {pocket_path}")
    for atom in pocket_rdkit.GetAtoms():
        atom.UpdatePropertyCache(strict=False)
    return Molecule.from_rdkit(ligand_rdkit), Molecule.from_rdkit(pocket_rdkit)


def molecule_graph(molecule, *, pocket: bool = False):
    conformer = molecule.GetConformer()
    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
    if not heavy_atoms:
        raise ValueError("Molecule has no heavy atoms")
    original_to_graph = {
        atom.GetIdx(): graph_index for graph_index, atom in enumerate(heavy_atoms)
    }
    graph = dgl.graph(([], []), num_nodes=len(heavy_atoms))
    graph.ndata["atom_feature"] = torch.tensor(
        [atom_features(atom) for atom in heavy_atoms], dtype=torch.float32
    )
    graph.ndata["atom_coordinate"] = torch.tensor(
        [
            [
                conformer.GetAtomPosition(atom.GetIdx()).x,
                conformer.GetAtomPosition(atom.GetIdx()).y,
                conformer.GetAtomPosition(atom.GetIdx()).z,
            ]
            for atom in heavy_atoms
        ],
        dtype=torch.float32,
    )
    graph.ndata["p_or_l"] = torch.full(
        (len(heavy_atoms), 1), float(pocket), dtype=torch.float32
    )
    graph.ndata["index"] = torch.tensor(
        [[atom.GetIdx()] for atom in heavy_atoms], dtype=torch.float32
    )
    graph.edata["edge_feature"] = torch.empty((0, 13), dtype=torch.float32)
    graph.edata["attention_weight"] = torch.empty((0, 1), dtype=torch.float32)
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in original_to_graph or end not in original_to_graph:
            continue
        feature = torch.tensor([edge_features(bond)], dtype=torch.float32)
        attention = torch.ones((1, 1), dtype=torch.float32)
        left, right = original_to_graph[begin], original_to_graph[end]
        graph.add_edges(left, right, {"edge_feature": feature, "attention_weight": attention})
        graph.add_edges(right, left, {"edge_feature": feature, "attention_weight": attention})
    return graph


def add_pocket_knn(graph, *, k: int = 2):
    if graph.num_nodes() <= 1 or k <= 0:
        return graph
    coordinates = graph.ndata["atom_coordinate"]
    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(float("inf"))
    k = min(int(k), graph.num_nodes() - 1)
    source = torch.arange(graph.num_nodes()).unsqueeze(1).expand(-1, k).reshape(-1)
    target = torch.topk(distances, k, largest=False, dim=1).indices.reshape(-1)
    not_self = source != target
    source, target = source[not_self].cpu(), target[not_self].cpu()
    missing = ~graph.has_edges_between(source, target)
    source, target = source[missing], target[missing]
    if source.numel() == 0:
        return graph
    features = torch.tensor(
        [edge_features()] * source.numel(), dtype=torch.float32
    )
    attention = torch.full((source.numel(), 1), 0.8, dtype=torch.float32)
    data = {"edge_feature": features, "attention_weight": attention}
    graph.add_edges(source, target, data)
    graph.add_edges(target, source, data)
    return graph


def merge_complex(ligand_graph, pocket_graph, *, cutoff: float = 6.0):
    distances = torch.cdist(
        ligand_graph.ndata["atom_coordinate"],
        pocket_graph.ndata["atom_coordinate"],
    )
    source, pocket_target = torch.where(distances < float(cutoff))
    graph = dgl.batch([ligand_graph, pocket_graph])
    target = pocket_target + ligand_graph.num_nodes()
    if source.numel():
        features = torch.tensor(
            [edge_features()] * source.numel(), dtype=torch.float32
        )
        attention = torch.full((source.numel(), 1), 0.8, dtype=torch.float32)
        data = {"edge_feature": features, "attention_weight": attention}
        graph.add_edges(source, target, data)
        graph.add_edges(target, source, data)
    return graph


def build_complex_graph(
    pocket_path: str | Path,
    ligand_path: str | Path,
    *,
    cutoff: float = 6.0,
    pocket_k: int = 2,
):
    pocket_path, ligand_path = Path(pocket_path), Path(ligand_path)
    if not pocket_path.is_file():
        raise FileNotFoundError(pocket_path)
    if not ligand_path.is_file():
        raise FileNotFoundError(ligand_path)
    ligand, pocket = _read_molecules(pocket_path, ligand_path)
    annotate_hbonds(ligand, pocket)
    ligand_graph = molecule_graph(ligand, pocket=False)
    pocket_graph = add_pocket_knn(molecule_graph(pocket, pocket=True), k=pocket_k)
    return merge_complex(ligand_graph, pocket_graph, cutoff=cutoff)


@dataclass(frozen=True)
class BuildRecord:
    uid: str
    status: str
    pocket_path: str
    ligand_path: str
    nodes: int | None = None
    edges: int | None = None
    error_type: str | None = None
    error: str | None = None


def _build_one(task):
    uid, pocket_path, ligand_path, cutoff, pocket_k = task
    try:
        graph = build_complex_graph(
            pocket_path, ligand_path, cutoff=cutoff, pocket_k=pocket_k
        )
        record = BuildRecord(
            uid, "success", str(pocket_path), str(ligand_path),
            graph.num_nodes(), graph.num_edges(),
        )
        return record, pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:  # failures are recorded and cause the entire build to fail
        record = BuildRecord(
            uid, "failed", str(pocket_path), str(ligand_path),
            error_type=type(error).__name__, error=str(error),
        )
        return record, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_lmdb(
    metadata_path: Path,
    output_path: Path,
    *,
    uid_col: str = "uid",
    pocket_col: str = "pocket_path",
    ligand_col: str = "ligand_path",
    cutoff: float = 6.0,
    pocket_k: int = 2,
    workers: int = 1,
    map_size: int = 1 << 40,
    transaction_size: int = 100,
) -> tuple[Path, Path]:
    """Create a new LMDB from explicit paths; relative paths use the CSV folder."""
    metadata_path, output_path = Path(metadata_path), Path(output_path)
    if workers < 1 or map_size < 1 or transaction_size < 1:
        raise ValueError("workers, map_size, and transaction_size must be positive")
    if not np.isfinite(cutoff) or cutoff <= 0 or pocket_k < 0:
        raise ValueError("cutoff must be positive and finite; pocket_k cannot be negative")
    metadata = pd.read_csv(metadata_path, dtype={uid_col: str, pocket_col: str, ligand_col: str})
    if metadata.empty:
        raise ValueError("Metadata contains no graph rows")
    required = {uid_col, pocket_col, ligand_col}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    for column in required:
        invalid = metadata[column].isna() | metadata[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Missing or empty values in {column!r}")
    if metadata[uid_col].duplicated().any():
        raise ValueError("Duplicate graph UIDs in metadata")
    rows_path = output_path.with_name(output_path.name + ".rows.csv")
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    for path in (output_path, rows_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in metadata.sort_values(uid_col).to_dict("records"):
        paths = [Path(row[pocket_col]), Path(row[ligand_col])]
        paths = [path if path.is_absolute() else metadata_path.parent / path for path in paths]
        tasks.append((row[uid_col], *paths, cutoff, pocket_k))
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = executor.map(_build_one, tasks, chunksize=16) if executor else map(_build_one, tasks)
    records = []
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    environment = lmdb.open(str(staging), map_size=map_size, subdir=True)
    transaction = environment.begin(write=True)
    writes = 0
    completed = False
    try:
        for record, payload in iterator:
            records.append(record)
            if payload is None:
                continue
            if not transaction.put(record.uid.encode("utf-8"), payload, overwrite=False):
                raise ValueError(f"Duplicate LMDB key: {record.uid}")
            writes += 1
            if writes % transaction_size == 0:
                transaction.commit()
                transaction = environment.begin(write=True)
        transaction.commit()
        transaction = None
        environment.sync()
        completed = True
    except Exception:
        if transaction is not None:
            transaction.abort()
        raise
    finally:
        environment.close()
        if executor is not None:
            executor.shutdown()
        if not completed or len(records) != len(tasks) or any(record.status != "success" for record in records):
            shutil.rmtree(staging, ignore_errors=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(rows_path, index=False)
    successful = sum(record.status == "success" for record in records)
    manifest = {
        "status": "complete" if successful == len(records) else "failed",
        "metadata_sha256": _sha256(metadata_path),
        "rows": len(records), "successful": successful,
        "failed": len(records) - successful,
        "parameters": {"cutoff": cutoff, "pocket_k": pocket_k},
        "software": {"torch": torch.__version__, "dgl": dgl.__version__,
                     "rdkit": rdkit.__version__,
                     "prolif": importlib.metadata.version("prolif")},
    }
    if successful == len(records):
        staging.rename(output_path)
        manifest["lmdb_data_sha256"] = _sha256(output_path / "data.mdb")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if successful != len(records):
        raise RuntimeError(f"Graph construction failed for {len(records) - successful} rows; see {rows_path}")
    return manifest_path, rows_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-lmdb", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--pocket-k", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--map-size", type=int, default=1 << 40)
    parser.add_argument("--transaction-size", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1 or args.transaction_size < 1 or args.map_size < 1:
        parser.error("workers, transaction-size, and map-size must be positive")
    manifest, rows = build_lmdb(
        args.metadata, args.output_lmdb,
        cutoff=args.cutoff,
        pocket_k=args.pocket_k, workers=args.workers, map_size=args.map_size,
        transaction_size=args.transaction_size,
    )
    print(f"manifest={manifest}")
    print(f"rows={rows}")


if __name__ == "__main__":
    main()
