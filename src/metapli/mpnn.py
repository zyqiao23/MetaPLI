"""Message-passing layers for protein–ligand graphs."""

from __future__ import annotations

import dgl.function as fn
import torch
from dgl.nn.functional import edge_softmax
from torch import nn


def get_activation(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU()
    if name == "LeakyReLU":
        return nn.LeakyReLU(0.2)
    if name == "ELU":
        return nn.ELU()
    if name == "GELU":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str | None = "ReLU",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.act = get_activation(activation) if activation is not None else None
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.fc(features)
        return self.act(output) if self.act is not None else output


class ResidualLayer(nn.Module):
    def __init__(self, hidden_dim: int, act: str = "ReLU") -> None:
        super().__init__()
        self.act = get_activation(act)
        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation=act)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=act)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.lin2(self.lin1(features))


class AngleEmbedding(nn.Module):
    def __init__(
        self, hidden_dim: int, class_num: int, activation: str = "ReLU"
    ) -> None:
        super().__init__()
        self.class_num = class_num
        self.angle_embedding_layer = nn.Embedding(class_num, hidden_dim)
        self.angle_input_layer = DenseLayer(
            hidden_dim, hidden_dim, activation, bias=True
        )

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        angle_class = (angle / (3.1415926 / self.class_num)).to(torch.int64)
        angle_class = angle_class.clamp(0, self.class_num - 1)
        return self.angle_input_layer(self.angle_embedding_layer(angle_class))


def _src_cat_edge(src_field: str, edge_field: str, out_field: str):
    def apply(edges):
        return {
            out_field: torch.cat(
                (edges.src[src_field], edges.data[edge_field]), dim=-1
            )
        }

    return apply


class AtomToEdge(nn.Module):
    def __init__(self, hidden_dim: int, activation: str = "ReLU") -> None:
        super().__init__()
        self.lin1 = DenseLayer(
            int(hidden_dim * 1.5), hidden_dim, activation=activation, bias=True
        )

    def forward(
        self, graph, atom_embedding: torch.Tensor, edge_embedding: torch.Tensor
    ) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata["h"] = atom_embedding
            graph.edata["h"] = edge_embedding
            graph.apply_edges(_src_cat_edge("h", "h", "h"))
            return self.lin1(graph.edata["h"])


class EdgeMessagePassing(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_head: int,
        feat_drop: float = 0,
        attn_drop: float = 0,
        activation: str = "ReLU",
        activation_att: str = "LeakyReLU",
        class_num: int = 6,
        dist_cutoff: int = 5,
    ) -> None:
        super().__init__()
        del dist_cutoff
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_feat = int(hidden_dim / num_head)
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn = nn.Parameter(torch.empty(1, num_head, self.out_feat))
        nn.init.xavier_normal_(self.attn)
        self.act_att = get_activation(activation_att)
        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=None)
        self.res1 = ResidualLayer(hidden_dim, activation)
        self.res2 = ResidualLayer(hidden_dim, activation)
        self.angle_embedding = AngleEmbedding(hidden_dim, class_num)
        self.angle1 = DenseLayer(hidden_dim, hidden_dim)
        self.angle2 = DenseLayer(hidden_dim, hidden_dim)
        self.act_func = get_activation(activation)

    def forward(
        self,
        graph,
        bond_embedding: torch.Tensor,
        index_kj: torch.Tensor,
        index_ji: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        idx_k: torch.Tensor,
    ) -> torch.Tensor:
        with graph.local_scope():
            position = graph.ndata["atom_coordinate"]
            pos_i = position[idx_i]
            pos_ji, pos_ki = position[idx_j] - pos_i, position[idx_k] - pos_i
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
            angle = torch.atan2(b, a)
            angle_embedding = self.angle_embedding(angle)
            angle_embedding = self.angle1(self.angle2(angle_embedding)).view(
                -1, self.num_head, self.out_feat
            )
            dist_decay = graph.edata["dist_decay"][index_kj]
            bond_embedding_feats = self.feat_drop(bond_embedding)
            feat_kj = self.k(bond_embedding_feats).view(
                -1, self.num_head, self.out_feat
            )[index_kj]
            feat_ji = self.q(bond_embedding_feats).view(
                -1, self.num_head, self.out_feat
            )[index_ji]
            feat = feat_kj + feat_ji + angle_embedding
            feat = self.act_att(feat)
            att = (feat * self.attn).sum(dim=-1).unsqueeze(dim=-1)
            att_decay = att + dist_decay.unsqueeze(dim=1)
            att_max, _ = torch.max(att_decay, dim=0, keepdim=True)
            att_decay = torch.exp(att_decay - att_max)
            att_all = torch.zeros(len(bond_embedding), self.num_head, 1).to(
                bond_embedding.device
            )
            att_all = att_all.index_add_(0, index_ji, att_decay)
            att_all = att_all[index_ji]
            att_all = torch.clamp(att_all, min=1e-9)
            att_decay = self.attn_drop(att_decay / att_all)
            v_att = (feat_kj * att_decay).view(-1, self.hidden_dim)
            v_clone = bond_embedding.clone()
            v_clone = v_clone.index_add_(0, index_ji, v_att) - v_clone
            he = self.lin2(self.lin1(v_clone))
            he = he + bond_embedding
            return self.res2(self.res1(he))


def _edge_product(edge_field1: str, edge_field2: str, out_field: str):
    def apply(edges):
        return {out_field: edges.data[edge_field1] * edges.data[edge_field2]}

    return apply


class EdgeToNode(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_head: int,
        feat_drop: float = 0,
        attn_drop: float = 0,
        activation: str = "ReLU",
        activation_att: str = "LeakyReLU",
        dist_cutoff: int = 5,
    ) -> None:
        super().__init__()
        del dist_cutoff
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_feat = int(hidden_dim / num_head)
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn = nn.Parameter(torch.empty(1, num_head, self.out_feat))
        nn.init.xavier_normal_(self.attn)
        self.act_att = get_activation(activation_att)
        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=None)
        self.res1 = ResidualLayer(hidden_dim, activation)
        self.res2 = ResidualLayer(hidden_dim, activation)

    def forward(
        self,
        graph,
        bond_embedding: torch.Tensor,
        atom_embedding: torch.Tensor,
        att_para: bool = False,
    ):
        with graph.local_scope():
            graph.edata["bond_embedding"] = bond_embedding.view(
                -1, self.num_head, self.out_feat
            )
            atom_h = self.feat_drop(atom_embedding)
            graph.ndata["k"] = self.k(atom_h).view(
                -1, self.num_head, self.out_feat
            )
            graph.ndata["q"] = self.q(atom_h).view(
                -1, self.num_head, self.out_feat
            )
            graph.apply_edges(fn.u_add_v("k", "q", "e"))
            att = self.act_att(graph.edata.pop("e"))
            att = (att * self.attn).sum(dim=-1).unsqueeze(dim=-1)
            att_decay = att + graph.edata["dist_decay"].unsqueeze(dim=1)
            att_decay = edge_softmax(graph, att_decay)
            graph.edata["att_decay"] = self.attn_drop(att_decay)
            graph.update_all(
                _edge_product("bond_embedding", "att_decay", "m"),
                fn.sum("m", "ft"),
            )
            he = graph.ndata["ft"].view(-1, self.hidden_dim)
            he = self.lin2(self.lin1(he))
            he = he + atom_embedding
            output = self.res2(self.res1(he))
            return (output, att_decay) if att_para else output


class MessagePassing(nn.Module):
    def __init__(
        self,
        num_head: int = 4,
        feat_drop: float = 0,
        attn_drop: float = 0,
        num_convs: int = 4,
        hidden_dim: int = 200,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.b2b_layers = nn.ModuleList()
        self.b2a_layers = nn.ModuleList()
        self.a2b = AtomToEdge(hidden_dim, activation=activation)
        for _ in range(num_convs):
            self.b2b_layers.append(
                EdgeMessagePassing(
                    hidden_dim,
                    num_head=num_head,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    activation=activation,
                )
            )
            self.b2a_layers.append(
                EdgeToNode(
                    hidden_dim,
                    num_head=num_head,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    activation=activation,
                )
            )

    def forward(
        self,
        graph,
        index_kj: torch.Tensor,
        index_ji: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        idx_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bond_embedding = self.a2b(
            graph, graph.ndata["atom_feature_h"], graph.edata["edge_feature_h"]
        )
        atom_embedding = graph.ndata["atom_feature_h"]
        for layer_num in range(self.num_convs - 1):
            bond_embedding = self.b2b_layers[layer_num](
                graph,
                bond_embedding,
                index_kj,
                index_ji,
                idx_i,
                idx_j,
                idx_k,
            )
            atom_embedding = self.b2a_layers[layer_num](
                graph, bond_embedding, atom_embedding
            )
        bond_embedding = self.b2b_layers[-1](
            graph, bond_embedding, index_kj, index_ji, idx_i, idx_j, idx_k
        )
        atom_embedding, attention = self.b2a_layers[-1](
            graph, bond_embedding, atom_embedding, att_para=True
        )
        return atom_embedding, attention
