import itertools

import numpy as np
import pandas as pd
import pytest

from metapli.context_policies import CONTEXT_POLICIES, PolicyContextIndex
from metapli.data import ContextIndex, EpisodeDataset


def metadata(n=8):
    return pd.DataFrame({"uid": [f"u{i:02d}" for i in range(n)],
                         "Uniprot": ["target"] * n, "pdb_id": ["source"] * n,
                         "Split": ["test"] * n,
                         "-logAffi": np.arange(n, dtype=float) + 1})


def similarities(n=8):
    return pd.DataFrame([{"pdb_id": "source", "uid1": f"u{i:02d}", "uid2": f"u{j:02d}",
                          "TanimotoSimilarity": 1 - j / (n + 1)}
                         for i, j in itertools.combinations(range(n), 2)])


@pytest.mark.parametrize("policy", CONTEXT_POLICIES)
def test_order_independence_repeatability_and_complete_split(policy):
    meta, sim = metadata(16), similarities(16)
    a = PolicyContextIndex(meta, sim, k=4, policy=policy, seed=2333)
    b = PolicyContextIndex(meta.sample(frac=1, random_state=1),
                           sim.sample(frac=1, random_state=2), k=4, policy=policy, seed=2333)
    assert len(EpisodeDataset(a, split="test")) == 16
    for uid in reversed(a.query_uids()):
        assert a.episode(uid) == b.episode(uid)
        assert a.episode(uid) == a.episode(uid)
        assert uid not in a.episode(uid).context_uids
        assert len(set(a.episode(uid).context_uids)) == 4


def test_topk_and_norm_rules_and_default_index():
    topk = ContextIndex(metadata(), similarities(), k=4)
    norm = PolicyContextIndex(metadata(), similarities(), k=4, policy="FPSimNorm")
    assert topk.episode("u00").context_uids == ("u01", "u02", "u03", "u04")
    assert norm.episode("u00").context_uids == ("u01", "u03", "u05", "u07")
    one = PolicyContextIndex(metadata(), similarities(), k=1, policy="FPSimNorm")
    assert one.episode("u00").context_uids == ("u01",)


@pytest.mark.parametrize("policy", CONTEXT_POLICIES)
def test_target_source_split_missing_activity_and_self_exclusion(policy):
    meta, sim = metadata(12), similarities(12)
    meta.loc[8, "Split"] = "train"
    meta.loc[9, "pdb_id"] = "different-source"
    meta.loc[10, "Uniprot"] = "different-target"
    meta.loc[11, "-logAffi"] = np.nan
    sim = pd.concat([sim, pd.DataFrame([{"pdb_id": "source", "uid1": "u00", "uid2": "u00",
                                        "TanimotoSimilarity": 1}])], ignore_index=True)
    for seed in range(10):
        index = PolicyContextIndex(meta, sim, k=4, policy=policy, seed=seed)
        assert set(index.episode("u00").context_uids) <= {f"u{i:02d}" for i in range(1, 8)}


@pytest.mark.parametrize("policy", CONTEXT_POLICIES)
@pytest.mark.parametrize("query_label", [np.nan, -300, 300])
def test_query_activity_does_not_control_selection(policy, query_label):
    meta, changed, sim = metadata(16), metadata(16), similarities(16)
    changed.loc[0, "-logAffi"] = query_label
    a = PolicyContextIndex(meta, sim, k=4, policy=policy, seed=23)
    b = PolicyContextIndex(changed, sim, k=4, policy=policy, seed=23)
    assert a.episode("u00").context_uids == b.episode("u00").context_uids
    if np.isnan(query_label):
        assert np.isnan(b.episode("u00").query_label)


def test_random_uses_entire_eligible_pool_and_epoch_seed_control():
    index = PolicyContextIndex(metadata(30), k=4, policy="Random", seed=23)
    first = index.episode("u00")
    index.episode("u01")
    assert first == index.episode("u00")
    index.set_epoch(1)
    assert first.context_uids != index.episode("u00").context_uids
    index.set_epoch(0)
    assert first == index.episode("u00")
    other = PolicyContextIndex(metadata(30), k=4, policy="Random", seed=2333)
    assert first.context_uids != other.episode("u00").context_uids
    seen = set()
    for epoch in range(100):
        index.set_epoch(epoch)
        seen.update(index.episode("u00").context_uids)
    assert seen == set(metadata(30).uid) - {"u00"}


def test_activity_inverse_pIC50_and_uniform_fallback():
    meta = metadata(5)
    meta["-logAffi"] = [7, 1, 2, 4, -1]
    index = PolicyContextIndex(meta, k=1, policy="ActivityBased", seed=23)
    for epoch in range(10):
        index.set_epoch(epoch)
        selected = index._rng(0).choice([1, 2, 3], size=1, replace=False,
                                       p=np.array([1, .5, .25]) / 1.75)
        assert index.episode("u00").context_uids == tuple(f"u{i:02d}" for i in selected)
    meta["-logAffi"] = [7, 1, 0, -1, -2]
    index = PolicyContextIndex(meta, k=3, policy="ActivityBased", seed=23)
    selected = index._rng(0).choice([1, 2, 3, 4], size=3, replace=False)
    assert index.episode("u00").context_uids == tuple(f"u{i:02d}" for i in selected)


@pytest.mark.parametrize("policy", CONTEXT_POLICIES)
def test_insufficient_contexts_fail_without_dropping_query(policy):
    index = PolicyContextIndex(metadata(4), similarities(4), k=4, policy=policy)
    assert len(index.query_uids()) == 4
    with pytest.raises(ValueError, match="exactly 4"):
        index.episode("u00")


@pytest.mark.parametrize("policy", ["FPSimTopk", "FPSimTopKExtra", "Custom", "topk", "random"])
def test_only_four_public_policy_names(policy):
    with pytest.raises(ValueError, match="Unknown context policy"):
        PolicyContextIndex(metadata(), similarities(), k=4, policy=policy)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_bad_k_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        PolicyContextIndex(metadata(), similarities(), k=k)


def test_similarity_duplicates_ties_and_conflicts():
    sim = similarities(5)
    sim["TanimotoSimilarity"] = .5
    duplicate = pd.concat([sim, sim.rename(columns={"uid1": "uid2", "uid2": "uid1"})])
    index = PolicyContextIndex(metadata(5), duplicate, k=4)
    assert index.episode("u00").context_uids == ("u01", "u02", "u03", "u04")
    conflict = sim.iloc[[0]].copy()
    conflict["TanimotoSimilarity"] = .6
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        PolicyContextIndex(metadata(5), pd.concat([sim, conflict]), k=4)


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.1, 1.01])
def test_invalid_similarity_fails(score):
    sim = similarities()
    sim.loc[0, "TanimotoSimilarity"] = score
    with pytest.raises(ValueError, match="finite Tanimoto"):
        PolicyContextIndex(metadata(), sim, k=4)


def test_chunked_csv_matches_memory_index_and_no_sim_random(tmp_path):
    meta, sim = metadata(12), similarities(12)
    meta.loc[8:, "Split"] = "train"
    meta.to_csv(tmp_path / "meta.csv", index=False)
    sim.to_csv(tmp_path / "sim.csv", index=False)
    index = PolicyContextIndex.from_csv(tmp_path / "meta.csv", tmp_path / "sim.csv",
                                        split="test", chunksize=3, k=4)
    reference = PolicyContextIndex(meta[meta.Split == "test"], sim, k=4)
    for uid in index.query_uids():
        assert index.episode(uid) == reference.episode(uid)
    random = PolicyContextIndex.from_csv(tmp_path / "meta.csv", tmp_path / "missing.csv",
                                         split="test", k=4, policy="Random")
    assert len(random.episode("u00").context_uids) == 4


def test_incomplete_similarity_fails_instead_of_partial_ranking():
    partial = similarities(8)
    partial = partial[~((partial.uid1 == "u00") & (partial.uid2 == "u07"))]
    index = PolicyContextIndex(metadata(8), partial, k=4)
    with pytest.raises(ValueError, match="incomplete"):
        index.episode("u00")


@pytest.mark.parametrize("label", ["invalid", np.inf, -np.inf])
def test_invalid_activity_fails_instead_of_becoming_unknown(label):
    meta = metadata()
    meta["-logAffi"] = meta["-logAffi"].astype(object)
    meta.loc[0, "-logAffi"] = label
    with pytest.raises(ValueError):
        PolicyContextIndex(meta, similarities(), k=4)
