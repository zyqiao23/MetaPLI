import json
from pathlib import Path

import lmdb
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from metapli.data import Episode, EpisodeCollator, LMDBGraphStore
from metapli.graph_builder import build_complex_graph, build_lmdb


def write_structures(root):
    root.mkdir(parents=True, exist_ok=True)
    ligand = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=11) == 0
    with Chem.SDWriter(str(root / "ligand.sdf")) as writer:
        writer.write(ligand)
    (root / "pocket.mol2").write_text(
        "@<TRIPOS>MOLECULE\nPOCKET\n3 2 1 0 0\nSMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 2.0 0.0 0.0 C.3 1 ALA1 0.0\n"
        "2 C2 3.5 0.0 0.0 C.3 1 ALA1 0.0\n"
        "3 N1 4.8 0.0 0.0 N.3 1 ALA1 0.0\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 2 3 1\n"
        "@<TRIPOS>SUBSTRUCTURE\n1 ALA1 1 GROUP 0 **** 0 ROOT\n"
    )


def test_prepared_structures_graph_store_and_collation(tmp_path):
    write_structures(tmp_path / "structures")
    graph = build_complex_graph(tmp_path / "structures/pocket.mol2",
                                tmp_path / "structures/ligand.sdf")
    assert graph.ndata["atom_feature"].shape == (6, 65)
    assert graph.edata["edge_feature"].shape[1] == 13
    assert graph.ndata["atom_coordinate"].shape == (6, 3)
    assert graph.ndata["p_or_l"].sum() == 3
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"uid": ["a", "b"], "pocket_path": ["structures/pocket.mol2"] * 2,
                  "ligand_path": ["structures/ligand.sdf"] * 2}).to_csv(metadata, index=False)
    output = tmp_path / "graphs.lmdb"
    manifest, rows = build_lmdb(metadata, output, map_size=16 * 1024 * 1024)
    assert json.loads(manifest.read_text())["status"] == "complete"
    assert pd.read_csv(rows).uid.tolist() == ["a", "b"]
    with LMDBGraphStore(output, cache_size=2) as store:
        loaded = store.get("a")
        assert loaded.batch_size == 1
        batch = EpisodeCollator(store)([Episode("a", ("b",), 7., (6.,), "source")])
        assert batch[0].batch_size == batch[1].batch_size == 1
        assert batch[2].tolist() == [7.]
        with pytest.raises(KeyError, match="missing"):
            store.get("unknown")


def test_failed_structure_is_reported_and_partial_lmdb_not_published(tmp_path):
    write_structures(tmp_path / "structures")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"uid": ["a", "b"],
                  "pocket_path": ["structures/pocket.mol2", "absent.mol2"],
                  "ligand_path": ["structures/ligand.sdf"] * 2}).to_csv(metadata, index=False)
    output = tmp_path / "graphs.lmdb"
    with pytest.raises(RuntimeError, match="failed for 1 rows"):
        build_lmdb(metadata, output, map_size=16 * 1024 * 1024)
    assert not output.exists()
    rows = pd.read_csv(tmp_path / "graphs.lmdb.rows.csv")
    assert rows.uid.tolist() == ["a", "b"]
    assert rows.status.tolist() == ["success", "failed"]
    assert rows.iloc[1].error_type == "FileNotFoundError"
    assert json.loads((tmp_path / "graphs.lmdb.manifest.json").read_text())["status"] == "failed"


def test_graph_builder_never_overwrites_existing_store(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"uid": ["a"], "pocket_path": ["p.mol2"], "ligand_path": ["l.sdf"]}).to_csv(metadata, index=False)
    output = tmp_path / "graphs.lmdb"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        build_lmdb(metadata, output)
