import pandas as pd
import pytest

from metapli.prepare_data import prepare_metadata
from metapli.similarity_builder import build_similarity


def records():
    return pd.DataFrame({"uid": ["c", "a", "b", "d"], "Uniprot": ["P"] * 4,
                         "pdb_id": ["s1", "s1", "s1", "s2"],
                         "-logAffi": [5., 6., 7., 8.], "Molecule SMILES": ["CC", "CCC", "CCCC", "CO"],
                         "pocket_path": ["pocket.mol2"] * 4, "ligand_path": ["ligand.sdf"] * 4})


def test_prepare_metadata_uses_complete_archived_split_and_keeps_records(tmp_path):
    source, splits, out = tmp_path / "records.csv", tmp_path / "split.csv", tmp_path / "metadata.csv"
    records().to_csv(source, index=False)
    pd.DataFrame({"uid": ["a", "b", "c", "d"], "Split": ["test"] * 4}).to_csv(splits, index=False)
    merged = prepare_metadata(source, splits, out)
    assert merged.uid.tolist() == ["a", "b", "c", "d"]
    assert merged["-logAffi"].tolist() == [6., 7., 5., 8.]
    assert set(merged.uid) == set(records().uid)
    assert merged.pocket_path.tolist() == ["pocket.mol2"] * 4


def test_prepare_metadata_missing_split_fails(tmp_path):
    source, splits = tmp_path / "records.csv", tmp_path / "split.csv"
    records().to_csv(source, index=False)
    pd.DataFrame({"uid": ["a", "b", "c"], "Split": ["test"] * 3}).to_csv(splits, index=False)
    with pytest.raises(ValueError, match="exactly the same UIDs"):
        prepare_metadata(source, splits, tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()


def test_similarity_contains_all_within_target_source_split_pairs(tmp_path):
    source = tmp_path / "meta.csv"
    frame = records()
    frame["Split"] = "test"
    frame.to_csv(source, index=False)
    result = build_similarity(source, tmp_path / "sim.csv")
    pairs = pd.read_csv(tmp_path / "sim.csv")
    assert result["pairs"] == 3
    assert set(zip(pairs.uid1, pairs.uid2)) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert pairs.TanimotoSimilarity.between(0, 1).all()
    frame.loc[frame.uid == "c", "Uniprot"] = "other"
    frame.to_csv(source, index=False)
    result = build_similarity(source, tmp_path / "sim2.csv")
    assert result["pairs"] == 1


def test_bad_smiles_fails_entire_similarity_build(tmp_path):
    frame = records()
    frame["Split"] = "test"
    frame.loc[0, "Molecule SMILES"] = "invalid-smiles"
    frame.to_csv(tmp_path / "meta.csv", index=False)
    with pytest.raises(ValueError, match="UIDs"):
        build_similarity(tmp_path / "meta.csv", tmp_path / "sim.csv")
    assert not (tmp_path / "sim.csv").exists()


def test_prepared_paths_are_relative_to_output_metadata(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    source, splits = folder / "records.csv", tmp_path / "split.csv"
    records().to_csv(source, index=False)
    pd.DataFrame({"uid": ["a", "b", "c", "d"], "Split": ["test"] * 4}).to_csv(splits, index=False)
    output = prepare_metadata(source, splits, tmp_path / "prepared/meta.csv")
    assert output.pocket_path.tolist() == ["../raw/pocket.mol2"] * 4


def test_archived_split_cannot_place_target_in_multiple_partitions(tmp_path):
    source, splits = tmp_path / "records.csv", tmp_path / "split.csv"
    records().to_csv(source, index=False)
    pd.DataFrame({"uid": ["a", "b", "c", "d"],
                  "Split": ["train", "train", "validation", "test"]}).to_csv(splits, index=False)
    with pytest.raises(ValueError, match="target cannot span"):
        prepare_metadata(source, splits, tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()
