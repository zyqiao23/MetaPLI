import pytest
import torch

from metapli.train import TrainingModule, _validate_config


def _config():
    return {"metadata_csv": "metadata.csv", "graph_lmdb": "graphs.lmdb", "output_dir": "output", "device": "cpu", "seed": 1, "context": {"k": 4, "policy": "FPSimTopK", "seed": 2}, "training": {"epochs": 1, "rank_size": 8}}


def test_listmle_gradient_matches_direct_objective():
    prediction = torch.tensor([[0.3], [-0.7], [1.1]], dtype=torch.float64, requires_grad=True)
    truth = torch.tensor([[7.0], [4.0], [8.0]], dtype=torch.float64)
    mask = torch.zeros_like(truth, dtype=torch.bool)
    actual = TrainingModule._listmle(prediction, truth, mask)
    order = prediction.reshape(-1)[torch.argsort(truth.reshape(-1), descending=True)]
    expected = (torch.flip(torch.logcumsumexp(torch.flip(order, [0]), 0), [0]) - order).mean()
    torch.testing.assert_close(actual, expected)
    value, gradient = TrainingModule._detached_ranking_gradient(prediction, truth, mask)
    torch.testing.assert_close(value, actual.detach())
    torch.testing.assert_close(gradient, torch.autograd.grad(actual, prediction)[0])


def test_singleton_ranking_group_has_finite_zero_gradient():
    prediction = torch.tensor([[0.5]], requires_grad=True)
    value, gradient = TrainingModule._detached_ranking_gradient(prediction, prediction.detach(), torch.zeros_like(prediction, dtype=torch.bool))
    assert value.item() == 0
    assert gradient.item() == 0


@pytest.mark.parametrize("key,value", [("limit", 1), ("target_subset", ["X"]), ("model", {"ablation": "anything"}), ("training", {"epochs": 1, "rank_size": 8, "query_allowlist": ["a"]})])
def test_training_rejects_unsupported_config(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError, match="unknown"):
        _validate_config(config)


def test_positive_ranking_weight_rejects_rank_size_one():
    config = _config()
    config["training"]["rank_size"] = 1
    with pytest.raises(ValueError, match="rank_size"):
        _validate_config(config)
