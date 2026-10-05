"""Synthetic end-to-end tests, independent of paper inputs and weights."""

import json
import pickle
from pathlib import Path

import dgl
import lmdb
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from metapli.evaluate import run as evaluate
from metapli.model import MetaPLI, default_model_config
from metapli.predict import run as predict
from metapli.train import run as train


def small_model_config():
    config = default_model_config()
    config["pocket_encoder_config"].update(hidden_dim=8, num_heads=2, radius=2, T=1, p_dropout=0.0)
    config["anp_config"].update(dim_in=8, dim_hid=8, dim_latent=8, heads=2, depth=2)
    config["context_num"] = 2
    return config


def _graph(rng):
    src = [i for i in range(4) for j in range(4) if i != j]
    dst = [j for i in range(4) for j in range(4) if i != j]
    graph = dgl.graph((src, dst), num_nodes=4)
    graph.ndata["atom_feature"] = torch.tensor(rng.normal(size=(4, 65)), dtype=torch.float32)
    graph.ndata["atom_coordinate"] = torch.tensor([[0, 0, 0], [1.1, 0, 0], [0, 1.2, 0], [0, 0, 1.3]], dtype=torch.float32)
    graph.ndata["p_or_l"] = torch.tensor([0, 0, 1, 1])
    graph.edata["edge_feature"] = torch.tensor(rng.normal(size=(12, 13)), dtype=torch.float32)
    graph.edata["attention_weight"] = torch.ones(12, 1)
    return graph


@pytest.fixture
def synthetic_inputs(tmp_path):
    torch.set_num_threads(1)
    records = []
    for split in ("train", "validation", "test"):
        for position in range(3):
            records.append({"uid": f"{split}_{position}", "Uniprot": f"protein_{split}", "pdb_id": f"source_{split}", "Split": split, "-logAffi": 5.0 + position})
    metadata = pd.DataFrame(records)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    pairs = []
    for source, group in metadata.groupby("pdb_id"):
        uids = group["uid"].tolist()
        for left in range(len(uids)):
            for right in range(left + 1, len(uids)):
                pairs.append({"pdb_id": source, "uid1": uids[left], "uid2": uids[right], "TanimotoSimilarity": 0.8 - right * 0.1})
    pd.DataFrame(pairs).to_csv(tmp_path / "similarities.csv", index=False)
    env = lmdb.open(str(tmp_path / "graphs.lmdb"), map_size=16 * 1024 * 1024)
    rng = np.random.default_rng(17)
    with env.begin(write=True) as transaction:
        for uid in metadata["uid"]:
            transaction.put(uid.encode(), pickle.dumps(_graph(rng)))
    env.close()
    torch.manual_seed(11)
    model = MetaPLI(small_model_config())
    torch.save({"state_dict": model.state_dict(), "model_config": model.config}, tmp_path / "model.ckpt")
    config = {"checkpoint": "model.ckpt", "metadata_csv": "metadata.csv", "graph_lmdb": "graphs.lmdb", "similarity_csv": "similarities.csv", "output_csv": "predictions.csv", "split": "test", "device": "cpu", "seed": 11, "context": {"k": 2, "policy": "FPSimTopK", "seed": 7}}
    return tmp_path, metadata, config


def _write(path, config):
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_train_predict_evaluate_processes_entire_splits(synthetic_inputs):
    root, metadata, predict_config = synthetic_inputs
    model = small_model_config()
    train_config = {"metadata_csv": "metadata.csv", "graph_lmdb": "graphs.lmdb", "similarity_csv": "similarities.csv", "output_dir": "training", "device": "cpu", "seed": 11, "context": {"k": 2, "policy": "Random", "seed": 7}, "model": {key: model[key] for key in ("pocket_encoder_config", "anp_config", "readout_scope")}, "training": {"epochs": 1, "rank_size": 2, "warmup_steps": 2}}
    original = MetaPLI(small_model_config())
    torch.manual_seed(11)
    original = MetaPLI(small_model_config())
    output = train(_write(root / "train.yaml", train_config))
    trained = MetaPLI.from_checkpoint(output / "best.ckpt")
    assert any(not torch.equal(value, trained.state_dict()[key]) for key, value in original.state_dict().items())
    history = pd.read_csv(output / "history.csv")
    assert history.loc[0, "train_queries"] == 3
    assert history.loc[0, "validation_queries"] == 3
    assert history.loc[0, "optimizer_steps"] == 2
    assert history.loc[0, "last_rank_group_size"] == 1
    contexts = [json.loads(line) for line in (output / "contexts.jsonl").read_text().splitlines()]
    assert {row["query_uid"] for row in contexts if row["stage"] == "train"} == set(metadata.loc[metadata.Split.eq("train"), "uid"])
    predict_config["checkpoint"] = "training/best.ckpt"
    predictions_path, manifest_path = predict(_write(root / "predict.yaml", predict_config))
    predictions = pd.read_csv(predictions_path)
    assert predictions["query_uid"].tolist() == ["test_0", "test_1", "test_2"]
    assert predictions.context_uids.str.split("#").map(len).eq(2).all()
    assert all(uid not in context.split("#") for uid, context in zip(predictions.query_uid, predictions.context_uids))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_queries"] == manifest["n_predictions"] == 3
    per_target, summary = evaluate(predictions_path, metadata_path=root / "metadata.csv")
    assert len(per_target) == 1
    assert summary["n_samples"] == 3
    assert np.isfinite(summary["macro_rmse"])


def test_prediction_accepts_unlabelled_query(synthetic_inputs):
    root, metadata, config = synthetic_inputs
    metadata.loc[metadata.uid.eq("test_0"), "-logAffi"] = np.nan
    metadata.to_csv(root / "metadata.csv", index=False)
    config["context"]["k"] = 1
    config["context"]["policy"] = "Random"
    output, _ = predict(_write(root / "predict.yaml", config))
    predictions = pd.read_csv(output)
    assert len(predictions) == 3
    assert predictions.loc[predictions.query_uid.eq("test_0"), "targets"].isna().all()
    assert predictions.preds.notna().all()
    with pytest.raises(ValueError, match="Non-finite"):
        evaluate(output, metadata_path=root / "metadata.csv")


@pytest.mark.parametrize("key,value", [("limit", 1), ("target_subset", ["protein_test"]), ("latent_finetune", {}), ("execution_authorized", True)])
def test_prediction_rejects_unsupported_config(synthetic_inputs, key, value):
    root, _, config = synthetic_inputs
    config[key] = value
    with pytest.raises(ValueError, match="unknown"):
        predict(_write(root / "predict.yaml", config))


@pytest.mark.parametrize("policy", ["FPSimTopK", "FPSimNorm", "Random", "ActivityBased"])
def test_all_four_context_policies_predict_all_queries(synthetic_inputs, policy):
    root, _, config = synthetic_inputs
    config["context"]["policy"] = policy
    output, _ = predict(_write(root / "predict.yaml", config))
    assert len(pd.read_csv(output)) == 3


def test_evaluation_rejects_incomplete_split(synthetic_inputs):
    root, _, config = synthetic_inputs
    output, _ = predict(_write(root / "predict.yaml", config))
    pd.read_csv(output).iloc[:2].to_csv(root / "incomplete.csv", index=False)
    with pytest.raises(ValueError, match="complete"):
        evaluate(root / "incomplete.csv", metadata_path=root / "metadata.csv")


def test_evaluation_requires_metadata(synthetic_inputs):
    root, _, config = synthetic_inputs
    output, _ = predict(_write(root / "predict.yaml", config))
    with pytest.raises(ValueError, match="Metadata is required"):
        evaluate(output, metadata_path=None)
