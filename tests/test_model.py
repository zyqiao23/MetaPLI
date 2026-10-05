"""Checkpoint, prediction and gradient checks on synthetic graph inputs."""

from copy import deepcopy

import dgl
import pytest
import torch

from metapli.anp import LatentEncoder
from metapli.model import MetaPLI, default_model_config
from metapli.mpnn import AngleEmbedding


def small_config():
    config = default_model_config()
    config["context_num"] = 2
    config["pocket_encoder_config"].update(
        hidden_dim=8, radius=2, T=1, num_heads=2, p_dropout=0.0
    )
    config["anp_config"].update(
        dim_in=8, dim_hid=8, dim_latent=8, depth=2, heads=2
    )
    return config


def graph(seed):
    generator = torch.Generator().manual_seed(seed)
    source = [i for i in range(4) for j in range(4) if i != j]
    destination = [j for i in range(4) for j in range(4) if i != j]
    result = dgl.graph((source, destination), num_nodes=4)
    result.ndata["atom_feature"] = torch.randn(4, 65, generator=generator)
    result.ndata["atom_coordinate"] = torch.randn(4, 3, generator=generator)
    result.ndata["p_or_l"] = torch.tensor([0, 0, 1, 1])
    result.edata["edge_feature"] = torch.randn(12, 13, generator=generator)
    result.edata["attention_weight"] = torch.full((12, 1), 0.5)
    return result


def inputs():
    return (
        dgl.batch([graph(1), graph(2), graph(3)]),
        dgl.batch([graph(4), graph(5), graph(6), graph(7)]),
        torch.tensor([[5.0, 6.0], [6.5, 7.0]]),
        torch.tensor([1, 0, 0], dtype=torch.long),
    )


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize("configuration_key", ["model_config", "hyper_parameters"])
def test_checkpoint_roundtrip_predictions(tmp_path, prefix, configuration_key):
    torch.manual_seed(10)
    model = MetaPLI(small_config()).eval()
    checkpoint = {
        "state_dict": {prefix + key: value for key, value in model.state_dict().items()},
        configuration_key: deepcopy(model.config),
    }
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    restored = MetaPLI.from_checkpoint(path).eval()
    assert restored.config == model.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    for latent_mode in ("sample", "prior_mean"):
        q_graph, c_graph, labels, tasks = inputs()
        torch.manual_seed(20)
        with torch.no_grad():
            expected = model(q_graph, c_graph, None, labels, tasks, latent_mode=latent_mode)
        q_graph, c_graph, labels, tasks = inputs()
        torch.manual_seed(20)
        with torch.no_grad():
            actual = restored(q_graph, c_graph, None, labels, tasks, latent_mode=latent_mode)
        for index in (1, 2, 5, 6, 8):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        assert actual[3] is None
        assert actual[7] is None


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_checkpoint_requires_all_parameter_keys(tmp_path, change):
    model = MetaPLI(small_config())
    state = dict(model.state_dict())
    if change == "missing":
        state.pop(next(iter(state)))
    else:
        state["extra.weight"] = torch.ones(1)
    path = tmp_path / "incomplete.ckpt"
    torch.save({"state_dict": state, "model_config": model.config}, path)
    with pytest.raises(RuntimeError):
        MetaPLI.from_checkpoint(path)


def test_training_posterior_and_gradients():
    torch.manual_seed(30)
    model = MetaPLI(small_config()).train()
    q_graph, c_graph, labels, tasks = inputs()
    truth = torch.tensor([7.2, 5.2, 5.8])
    output = model(q_graph, c_graph, truth, labels, tasks)
    prediction, mu, sigma, posterior, prior = output[:5]
    dense_truth, padding = output[7:]
    assert posterior is not None
    assert tuple(mu.shape) == (2, 2, 1)
    assert torch.equal(padding, torch.tensor([[False, False], [False, True]]))
    torch.testing.assert_close(dense_truth[0], truth[1:])
    torch.testing.assert_close(dense_truth[1, 0], truth[0])
    assert torch.isfinite(mu).all()
    assert (sigma > 0).all()
    nll = -prediction.log_prob(dense_truth.unsqueeze(-1)).squeeze(-1)[~padding].mean()
    kl = torch.distributions.kl_divergence(posterior, prior).mean()
    (nll + kl).backward()
    for module in (
        model.pocket_encoder, model.latent_encoder,
        model.deterministic_encoder, model.decoder,
    ):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0


def test_evaluation_rejects_query_labels():
    model = MetaPLI(small_config()).eval()
    q_graph, c_graph, labels, tasks = inputs()
    with pytest.raises(ValueError, match="only in training"):
        model(q_graph, c_graph, torch.ones(3), labels, tasks)


def test_prediction_checks_context_count():
    model = MetaPLI(small_config()).eval()
    q_graph, _, labels, tasks = inputs()
    with pytest.raises(ValueError, match="Expected 4 context graphs"):
        model(q_graph, dgl.batch([graph(4), graph(5)]), None, labels, tasks)


def test_prediction_checks_latent_mode():
    model = MetaPLI(small_config()).eval()
    q_graph, c_graph, labels, tasks = inputs()
    with pytest.raises(ValueError, match="Unknown latent_mode"):
        model(q_graph, c_graph, None, labels, tasks, latent_mode="unknown")


def test_collinear_angle_uses_final_bin_without_overflow():
    layer = AngleEmbedding(hidden_dim=8, class_num=6)
    angle = torch.atan2(torch.tensor([0.0]), torch.tensor([-1.0]))
    result = layer(angle)
    assert result.shape == (1, 8)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, layer(torch.tensor([3.0])))
    result.sum().backward()
    assert layer.angle_embedding_layer.weight.grad is not None


@pytest.mark.parametrize("padding_count", [1, 4])
def test_latent_encoder_padding_does_not_change_posterior(padding_count):
    torch.manual_seed(13)
    encoder = LatentEncoder(8, 8, 8, depth=2, heads=2, dim_label=1).eval()
    context_x = torch.randn(1, 2, 8)
    context_y = torch.randn(1, 2, 1)
    padded_x = torch.cat([context_x, 100 * torch.randn(1, padding_count, 8)], dim=1)
    padded_y = torch.cat([context_y, 100 * torch.randn(1, padding_count, 1)], dim=1)
    batch_x = torch.cat([padded_x, torch.randn_like(padded_x)], dim=0)
    batch_y = torch.cat([padded_y, torch.randn_like(padded_y)], dim=0)
    mask = torch.zeros(2, 2 + padding_count, dtype=torch.bool)
    mask[0, 2:] = True
    with torch.no_grad():
        _, expected_mu, expected_sigma = encoder(context_x, context_y)
        _, actual_mu, actual_sigma = encoder(batch_x, batch_y, mask)
    torch.testing.assert_close(actual_mu[:1], expected_mu, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_sigma[:1], expected_sigma, rtol=1e-6, atol=1e-7)


def test_latent_encoder_rejects_all_padding_task():
    encoder = LatentEncoder(8, 8, 8, depth=2, heads=2, dim_label=1)
    with pytest.raises(ValueError, match="at least one unmasked context"):
        encoder(
            torch.zeros(1, 2, 8),
            torch.zeros(1, 2, 1),
            torch.ones(1, 2, dtype=torch.bool),
        )
