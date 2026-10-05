"""MPNN and ANP model for context-conditioned activity prediction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from .anp import Decoder, DeterministicEncoder, LatentEncoder
from .pocket_encoder import MPNN

LatentMode = Literal["sample", "prior_mean"]


def default_model_config() -> dict[str, object]:
    return {
        "pocket_encoder_config": {
            "hidden_dim": 96,
            "radius": 3,
            "T": 2,
            "p_dropout": 0.1,
            "num_heads": 4,
            "in_dim_atom": 65,
            "in_dim_edge": 13,
        },
        "anp_config": {
            "dim_in": 96,
            "dim_hid": 96,
            "dim_latent": 96,
            "depth": 3,
            "heads": 4,
            "dim_label": 1,
            "dim_out": 1,
        },
        "context_num": 4,
        "readout_scope": "lig",
    }


class MetaPLI(nn.Module):
    """Encode protein–ligand graphs with MPNN and predict activities with ANP."""

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__()
        self.config = deepcopy(config or default_model_config())
        pocket_config = dict(self.config["pocket_encoder_config"])
        anp_config = dict(self.config["anp_config"])
        self.context_num = int(self.config.get("context_num", 4))
        self.readout_scope = str(self.config.get("readout_scope", "lig"))

        self.pocket_encoder = MPNN(pocket_config)
        self.latent_encoder = LatentEncoder(
            int(anp_config["dim_in"]),
            int(anp_config["dim_hid"]),
            int(anp_config["dim_latent"]),
            int(anp_config["depth"]),
            int(anp_config["heads"]),
            int(anp_config["dim_label"]),
        )
        self.deterministic_encoder = DeterministicEncoder(
            int(anp_config["dim_in"]),
            int(anp_config["dim_hid"]),
            int(anp_config["dim_latent"]),
            int(anp_config["depth"]),
            int(anp_config["heads"]),
            int(anp_config["dim_label"]),
        )
        self.decoder = Decoder(
            int(anp_config["dim_in"]),
            int(anp_config["dim_hid"]),
            int(anp_config["dim_latent"]),
            int(anp_config["depth"]),
            int(anp_config["dim_out"]),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        context_num: int | None = None,
    ) -> "MetaPLI":
        """Load a complete model checkpoint without dropping any parameters.

        Native checkpoints store ``model_config`` and ``state_dict``. Training
        framework checkpoints may instead store configuration fields under
        ``hyper_parameters`` and prefix the parameter keys with ``model.``.
        """
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain a 'state_dict'")
        config = deepcopy(checkpoint.get("model_config", default_model_config()))
        if "model_config" not in checkpoint:
            parameters = checkpoint.get("hyper_parameters", {})
            for key in (
                "pocket_encoder_config", "anp_config", "context_num", "readout_scope"
            ):
                if key in parameters:
                    config[key] = parameters[key]
        if context_num is not None:
            config["context_num"] = int(context_num)
        model = cls(config)
        state_dict = checkpoint["state_dict"]
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {key[len("model."):]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _pack_queries(
        query_features: torch.Tensor,
        query_labels: torch.Tensor | None,
        task_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        unique_tasks, counts = torch.unique(task_indices, sorted=True, return_counts=True)
        if unique_tasks.numel() == 0:
            raise ValueError("A batch must contain at least one task")
        max_queries = int(counts.max().item())
        dense_features = query_features.new_zeros(
            unique_tasks.numel(), max_queries, query_features.shape[-1]
        )
        dense_labels = None
        if query_labels is not None:
            dense_labels = query_labels.new_zeros(unique_tasks.numel(), max_queries)
        padding_mask = torch.ones(
            unique_tasks.numel(),
            max_queries,
            dtype=torch.bool,
            device=query_features.device,
        )
        for row, task in enumerate(unique_tasks):
            selected = task_indices == task
            count = int(selected.sum().item())
            dense_features[row, :count] = query_features[selected]
            if dense_labels is not None:
                dense_labels[row, :count] = query_labels[selected]
            padding_mask[row, :count] = False
        return dense_features, dense_labels, padding_mask

    def forward(
        self,
        query_graph,
        context_graph,
        query_labels: torch.Tensor | None,
        context_labels: torch.Tensor,
        task_indices: torch.Tensor,
        *,
        latent_mode: LatentMode = "sample",
    ):
        """Return the predictive Normal distribution and training intermediates.

        Context labels condition every prediction. Query labels are allowed
        only in training mode to construct the posterior used by the ANP
        objective. In evaluation mode, predictions use the context prior.

        Context graphs are ordered by sorted task ID, followed by their K
        context records. Output query rows use the same sorted task order.
        """
        if query_labels is not None and not self.training:
            raise ValueError("query_labels are allowed only in training mode")
        if latent_mode not in {"sample", "prior_mean"}:
            raise ValueError(f"Unknown latent_mode: {latent_mode}")
        query_features, _ = self.pocket_encoder(
            query_graph, self.readout_scope
        )
        context_features, _ = self.pocket_encoder(
            context_graph, self.readout_scope
        )
        dense_query, dense_labels, query_mask = self._pack_queries(
            query_features, query_labels, task_indices
        )
        num_tasks = dense_query.shape[0]
        expected_contexts = num_tasks * self.context_num
        if context_features.shape[0] != expected_contexts:
            raise ValueError(
                f"Expected {expected_contexts} context graphs for {num_tasks} tasks "
                f"and K={self.context_num}, got {context_features.shape[0]}"
            )
        if tuple(context_labels.shape) != (num_tasks, self.context_num):
            raise ValueError(
                "context_labels must have shape "
                f"({num_tasks}, {self.context_num}), got {tuple(context_labels.shape)}"
            )
        dense_context = context_features.reshape(
            num_tasks, self.context_num, context_features.shape[-1]
        )
        context_y = context_labels.unsqueeze(-1)
        prior, prior_mu, _ = self.latent_encoder(dense_context, context_y)

        if query_labels is not None and self.training:
            if dense_labels is None:
                raise AssertionError("Dense labels unexpectedly missing")
            posterior, _, _ = self.latent_encoder(
                dense_query, dense_labels.unsqueeze(-1), query_mask
            )
            z = posterior.rsample()
        else:
            posterior = None
            if latent_mode == "sample":
                z = prior.sample()
            elif latent_mode == "prior_mean":
                z = prior_mu
            else:
                raise ValueError(f"Unknown latent_mode: {latent_mode}")

        z_expanded = z.unsqueeze(1).expand(-1, dense_query.shape[1], -1)
        deterministic = self.deterministic_encoder(
            dense_context, context_y, dense_query, query_mask
        )
        distribution, mu, sigma = self.decoder(
            deterministic, z_expanded, dense_query
        )
        return (
            distribution,
            mu,
            sigma,
            posterior,
            prior,
            z,
            deterministic,
            dense_labels,
            query_mask,
        )
