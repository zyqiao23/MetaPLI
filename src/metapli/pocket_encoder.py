"""Protein--ligand graph encoder used by MetaPLI."""

from __future__ import annotations

import dgl
import torch
from dgllife.model.readout import AttentiveFPReadout as AttentionReadout
from torch import nn
from torch_sparse import SparseTensor

from .mpnn import MessagePassing


class DenseLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(features))


def _distance(edge_feat: str):
    def apply(edges):
        return {
            "dist": (edges.src[edge_feat] - edges.dst[edge_feat])
            .pow(2)
            .sum(dim=-1)
            .sqrt()
        }

    return apply


class MPNN(nn.Module):
    def __init__(self, pocket_encoder_config: dict[str, int | float]) -> None:
        super().__init__()
        hidden_dim = int(pocket_encoder_config["hidden_dim"])
        self.encoder = MessagePassing(
            num_head=int(pocket_encoder_config["num_heads"]),
            feat_drop=float(pocket_encoder_config["p_dropout"]),
            attn_drop=float(pocket_encoder_config["p_dropout"]),
            num_convs=int(pocket_encoder_config["radius"]),
            hidden_dim=hidden_dim,
            activation="ReLU",
        )
        self.readout = AttentionReadout(
            feat_size=hidden_dim,
            num_timesteps=int(pocket_encoder_config["T"]),
            dropout=float(pocket_encoder_config["p_dropout"]),
        )
        self.lin_atom1 = DenseLayer(
            int(pocket_encoder_config["in_dim_atom"]), hidden_dim, bias=True
        )
        self.lin_edge1 = DenseLayer(
            int(pocket_encoder_config["in_dim_edge"]), hidden_dim // 2, bias=True
        )

    @staticmethod
    def triplets(graph):
        row, col = graph.edges()
        values = torch.arange(row.size(0), device=row.device)
        adjacency = SparseTensor(
            row=col,
            col=row,
            value=values,
            sparse_sizes=(graph.num_nodes(), graph.num_nodes()),
        )
        adjacency_rows = adjacency[row]
        counts = adjacency_rows.set_value(None).sum(dim=1).to(torch.long)
        idx_i = col.repeat_interleave(counts)
        idx_j = row.repeat_interleave(counts)
        idx_k = adjacency_rows.storage.col()
        mask = idx_i != idx_k
        return (
            col,
            row,
            idx_i[mask],
            idx_j[mask],
            idx_k[mask],
            adjacency_rows.storage.value()[mask],
            adjacency_rows.storage.row()[mask],
        )

    def forward(self, graph, readout_scope: str = "all"):
        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(graph)
        graph.ndata["atom_feature_h"] = self.lin_atom1(
            graph.ndata["atom_feature"].to(torch.float32)
        )
        graph.edata["edge_feature_h"] = self.lin_edge1(
            graph.edata["edge_feature"].to(torch.float32)
        )
        graph.apply_edges(_distance("atom_coordinate"))
        raw_attention = torch.where(
            graph.edata["attention_weight"] == 0,
            torch.tensor(-1.0, device=graph.device),
            graph.edata["attention_weight"].to(torch.float32),
        )
        graph.edata["dist_decay"] = torch.where(
            graph.edata["attention_weight"] == 1,
            torch.tensor(1.0, device=graph.device),
            raw_attention,
        )
        hidden, attention = self.encoder(
            graph, idx_kj, idx_ji, idx_i, idx_j, idx_k
        )

        with graph.local_scope():
            graph.ndata["h"] = hidden
            if readout_scope == "all":
                selected_graph = graph
                selected_hidden = hidden
            elif readout_scope in {"lig", "poc"}:
                role = 0 if readout_scope == "lig" else 1
                role_mask = graph.ndata["p_or_l"] == role
                nodes = torch.nonzero(role_mask, as_tuple=True)[0]
                if nodes.numel() == 0:
                    raise ValueError(f"Graph has no nodes for readout scope {readout_scope!r}")
                selected_graph = dgl.node_subgraph(graph, nodes)
                batch_sizes = graph.batch_num_nodes().tolist()
                selected_counts = torch.tensor(
                    [int(part.sum().item()) for part in torch.split(role_mask, batch_sizes)],
                    dtype=graph.batch_num_nodes().dtype,
                    device=graph.device,
                )
                if (selected_counts == 0).any():
                    raise ValueError(
                        f"At least one graph has no nodes for readout scope {readout_scope!r}"
                    )
                selected_graph.set_batch_num_nodes(selected_counts)
                selected_hidden = hidden[nodes]
            else:
                raise ValueError("readout_scope must be 'all', 'lig', or 'poc'")
            return self.readout(
                selected_graph, selected_hidden, get_node_weight=False
            ), attention
