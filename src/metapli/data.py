"""Explicit metadata, context retrieval, and graph-store interfaces."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

import lmdb
import torch
from torch.utils.data import Dataset

import dgl

@dataclass(frozen=True)
class Episode:
    query_uid: str
    context_uids: tuple[str, ...]
    query_label: float
    context_labels: tuple[float, ...]
    group_id: str


class ContextIndex:
    """Convenience constructor for the manuscript FPSimTopK context policy."""

    def __new__(cls, *args, **kwargs):
        from .context_policies import PolicyContextIndex
        return PolicyContextIndex(*args, **kwargs)

    @classmethod
    def from_csv(cls, *args, **kwargs):
        from .context_policies import PolicyContextIndex
        return PolicyContextIndex.from_csv(*args, **kwargs)


class LMDBGraphStore:
    """Read a single immutable graph store and validate the model input schema."""

    def __init__(self, path: str | Path, *, cache_size: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self.cache_size = cache_size
        self._cache: OrderedDict[str, object] = OrderedDict()
        self.environment = lmdb.open(
            str(self.path), readonly=True, lock=False, readahead=False,
            max_readers=256,
        )

    def get(self, uid: str):
        uid = str(uid)
        if uid in self._cache:
            graph = self._cache.pop(uid)
            self._cache[uid] = graph
            return graph
        with self.environment.begin(write=False) as transaction:
            payload = transaction.get(uid.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Graph UID {uid!r} is missing from {self.path}")
        graph = pickle.loads(payload)
        required_nodes = {"atom_feature", "atom_coordinate", "p_or_l"}
        required_edges = {"edge_feature", "attention_weight"}
        missing_nodes = sorted(required_nodes - set(graph.ndata))
        missing_edges = sorted(required_edges - set(graph.edata))
        if missing_nodes or missing_edges:
            raise ValueError(
                f"Graph {uid!r} violates schema: missing node={missing_nodes}, "
                f"edge={missing_edges}"
            )
        if graph.ndata["atom_feature"].shape[-1] != 65:
            raise ValueError(f"Graph {uid!r} does not have 65-dimensional atoms")
        if graph.edata["edge_feature"].shape[-1] != 13:
            raise ValueError(f"Graph {uid!r} does not have 13-dimensional edges")
        # One LMDB key describes one complex, including its ligand and pocket.
        graph.set_batch_num_nodes(
            torch.tensor([graph.num_nodes()], dtype=graph.batch_num_nodes().dtype)
        )
        graph.set_batch_num_edges(
            torch.tensor([graph.num_edges()], dtype=graph.batch_num_edges().dtype)
        )
        if self.cache_size:
            self._cache[uid] = graph
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return graph

    def close(self) -> None:
        self._cache.clear()
        self.environment.close()

    def __enter__(self) -> "LMDBGraphStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class EpisodeDataset(Dataset):
    def __init__(self, index, *, split: str) -> None:
        self.index = index
        self.uids = index.query_uids(split=split)
        if not self.uids:
            raise ValueError(f"No query UIDs found for split {split!r}")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, item: int) -> Episode:
        return self.index.episode(self.uids[item])


class EpisodeCollator:
    """Build a fail-fast graph batch; use with ``num_workers=0``."""

    def __init__(self, store: LMDBGraphStore) -> None:
        self.store = store

    def __call__(self, episodes: list[Episode]):
        if not episodes:
            raise ValueError("Cannot collate an empty episode batch")
        k_values = {len(episode.context_uids) for episode in episodes}
        if len(k_values) != 1:
            raise ValueError(f"Context sizes differ within batch: {sorted(k_values)}")
        query_graphs = [self.store.get(episode.query_uid) for episode in episodes]
        context_graphs = [
            self.store.get(uid)
            for episode in episodes
            for uid in episode.context_uids
        ]
        return (
            dgl.batch(query_graphs),
            dgl.batch(context_graphs),
            torch.tensor([episode.query_label for episode in episodes], dtype=torch.float32),
            torch.tensor(
                [episode.context_labels for episode in episodes], dtype=torch.float32
            ),
            torch.arange(len(episodes), dtype=torch.long),
            [episode.query_uid for episode in episodes],
            [episode.context_uids for episode in episodes],
        )
