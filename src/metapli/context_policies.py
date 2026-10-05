"""Four manuscript context policies for leave-one-query-out conditioning.

Eligible contexts have an observed activity and the same target, protein
structure source and partition as the query. FPSimTopK uses ECFP4/Tanimoto
ranking; FPSimNorm spans equally spaced ranking positions; Random samples
without replacement; ActivityBased samples with inverse positive pIC50 weights
and uses uniform sampling if there are too few positive context labels.

Query activity is never used to choose contexts. Query UIDs are not removed
when their context pool is too small: episode construction raises an error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Episode


_SIM_COLUMNS = ("pdb_id", "uid1", "uid2", "TanimotoSimilarity")
CONTEXT_POLICIES = ("FPSimTopK", "FPSimNorm", "Random", "ActivityBased")


class _NeighbourMapping(Mapping):
    """Lazy compatibility view; do not materialize millions of UID strings."""

    def __init__(self, owner: "PolicyContextIndex") -> None:
        self.owner = owner

    def __iter__(self):
        for index, uid in enumerate(self.owner._uids):
            if self.owner._offsets[index] != self.owner._offsets[index + 1]:
                yield (str(self.owner._groups[index]), str(uid))

    def __len__(self):
        return int(np.count_nonzero(np.diff(self.owner._offsets)))

    def __getitem__(self, key):
        group, uid = key
        index = self.owner._uid_to_index.get(str(uid))
        if index is None or str(self.owner._groups[index]) != str(group):
            raise KeyError(key)
        start, stop = self.owner._offsets[index:index + 2]
        if start == stop:
            raise KeyError(key)
        return tuple(str(self.owner._uids[i]) for i in self.owner._neighbour_indices[start:stop])


class PolicyContextIndex:
    """A ContextIndex-compatible, reproducible policy-aware retrieval index.

    ``seed`` is the retrieval seed, distinct from model-sampling seeds.
    Epoch defaults to zero. Training calls ``set_epoch(epoch)`` once per epoch;
    validation/testing use a separate index fixed at epoch zero. Changing CSV
    order, query visitation order, batch size, or worker count cannot change an
    episode. Independent retrieval seeds can coincidentally select the same set.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        similarities: pd.DataFrame | None = None,
        *,
        k: int,
        policy: str = "FPSimTopK",
        seed: int = 0,
        uid_col: str = "uid",
        group_col: str = "pdb_id",
        label_col: str = "-logAffi",
        split_col: str = "Split",
        target_col: str = "Uniprot",
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if policy not in CONTEXT_POLICIES:
            raise ValueError(f"Unknown context policy: {policy!r}; use {CONTEXT_POLICIES}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        required = {uid_col, group_col, target_col, label_col, split_col}
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError(f"Metadata is missing columns: {missing}")
        for column in (uid_col, group_col, target_col, split_col):
            invalid = metadata[column].isna() | (metadata[column].astype(str).str.strip() == "")
            if invalid.any():
                raise ValueError(f"Metadata has missing/empty identifiers in {column!r}")
        duplicate_mask = metadata[uid_col].astype(str).duplicated()
        if duplicate_mask.any():
            raise ValueError(f"Duplicate metadata UIDs: {metadata.loc[duplicate_mask, uid_col].head().tolist()}")

        self.k, self.policy = int(k), policy
        self.seed, self.epoch = int(seed), 0
        self.uid_col, self.group_col = uid_col, group_col
        self.label_col, self.split_col, self.target_col = label_col, split_col, target_col
        self.metadata = metadata.copy()
        for column in (uid_col, group_col, target_col, split_col):
            self.metadata[column] = self.metadata[column].astype(str)
        self.metadata[label_col] = pd.to_numeric(self.metadata[label_col], errors="raise")
        if np.isinf(self.metadata[label_col].to_numpy(dtype=float)).any():
            raise ValueError("Activities must be finite or missing")
        self.rows = self.metadata.set_index(uid_col, drop=False)
        ordered = self.metadata.sort_values(uid_col, kind="stable")
        self._uids = ordered[uid_col].to_numpy(dtype=object)
        self._groups = ordered[group_col].to_numpy(dtype=object)
        self._targets = ordered[target_col].to_numpy(dtype=object)
        self._splits = ordered[split_col].to_numpy(dtype=object)
        self._labels = ordered[label_col].to_numpy(dtype=float)
        self._finite = np.isfinite(self._labels)
        self._uid_to_index = {str(uid): index for index, uid in enumerate(self._uids)}
        if len(self._uids) > np.iinfo(np.int32).max:
            raise ValueError("Too many UIDs for the compact int32 retrieval index")
        pools: dict[tuple[str, str, str], list[int]] = {}
        for index in np.flatnonzero(self._finite):
            key = (str(self._targets[index]), str(self._groups[index]), str(self._splits[index]))
            pools.setdefault(key, []).append(int(index))
        self._pools = {key: np.asarray(values, dtype=np.int32) for key, values in pools.items()}
        self._offsets = np.zeros(len(self._uids) + 1, dtype=np.int64)
        self._neighbour_indices = np.empty(0, dtype=np.int32)
        self.neighbours = _NeighbourMapping(self)
        if self._needs_similarities:
            if similarities is None:
                raise ValueError(f"{self.policy} with k>0 requires similarities")
            self._build_csr(lambda: iter((similarities,)))

    @property
    def _needs_similarities(self) -> bool:
        return self.policy in ("FPSimTopK", "FPSimNorm")

    @classmethod
    def from_csv(
        cls,
        metadata_path: str | Path,
        similarity_path: str | Path | None = None,
        *,
        split: str | None = None,
        chunksize: int = 250_000,
        **kwargs,
    ) -> "PolicyContextIndex":
        """Read metadata and a bounded-memory similarity index.

        ``split`` optionally filters metadata before retrieval construction.
        Random and ActivityBased require metadata only. Similarity-dependent
        runs scan only the four required columns and retain eligible known UID
        pairs from the selected metadata. Input files must remain immutable
        during the two-pass scan.
        """
        if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
            raise ValueError("chunksize must be a positive integer")
        uid_col, group_col = kwargs.get("uid_col", "uid"), kwargs.get("group_col", "pdb_id")
        target_col = kwargs.get("target_col", "Uniprot")
        split_col = kwargs.get("split_col", "Split")
        metadata = pd.read_csv(metadata_path, dtype={uid_col: str, group_col: str, target_col: str, split_col: str})
        if split is not None:
            if split_col not in metadata.columns:
                raise ValueError(f"Metadata is missing columns: {[split_col]}")
            metadata = metadata[metadata[split_col].astype(str) == str(split)].copy()
        policy = kwargs.get("policy", "FPSimTopK")
        needs_sim = policy in ("FPSimTopK", "FPSimNorm")
        # An empty schema-bearing frame delays the actual similarity build to
        # the chunked CSV path without changing public constructor validation.
        index = cls(metadata, pd.DataFrame(columns=_SIM_COLUMNS) if needs_sim else None, **kwargs)
        if needs_sim:
            if similarity_path is None:
                raise ValueError(f"{index.policy} with k>0 requires a similarity CSV")
            header = pd.read_csv(similarity_path, nrows=0)
            missing = sorted(set(_SIM_COLUMNS) - set(header.columns))
            if missing:
                raise ValueError(f"Similarity table is missing columns: {missing}")

            def chunks() -> Iterator[pd.DataFrame]:
                yield from pd.read_csv(
                    similarity_path, usecols=list(_SIM_COLUMNS),
                    dtype={"pdb_id": str, "uid1": str, "uid2": str}, chunksize=chunksize,
                )

            index._build_csr(chunks)
        return index

    def _filtered_pairs(self, chunks: Iterable[pd.DataFrame]):
        for chunk in chunks:
            missing = sorted(set(_SIM_COLUMNS) - set(chunk.columns))
            if missing:
                raise ValueError(f"Similarity table is missing columns: {missing}")
            left = chunk["uid1"].astype(str).map(self._uid_to_index)
            right = chunk["uid2"].astype(str).map(self._uid_to_index)
            known = left.notna() & right.notna()
            if not known.any():
                continue
            a, b = left[known].to_numpy(dtype=np.int32), right[known].to_numpy(dtype=np.int32)
            groups = chunk.loc[known, "pdb_id"].astype(str).to_numpy()
            eligible = (
                (a != b) & (self._groups[a] == groups) & (self._groups[b] == groups)
                & (self._splits[a] == self._splits[b]) & (self._targets[a] == self._targets[b])
            )
            if not eligible.any():
                continue
            scores = pd.to_numeric(chunk.loc[known, "TanimotoSimilarity"], errors="coerce").to_numpy(dtype=float)
            a, b, scores = a[eligible], b[eligible], scores[eligible]
            if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
                raise ValueError("Eligible similarity pairs require finite Tanimoto scores in [0, 1]")
            yield a, b, scores

    def _build_csr(self, chunks_factory) -> None:
        """Two-pass counts/fill, then per-row deduplication and rank sorting."""
        size = len(self._uids)
        counts = np.zeros(size, dtype=np.int64)
        for left, right, _ in self._filtered_pairs(chunks_factory()):
            counts += np.bincount(left, minlength=size) + np.bincount(right, minlength=size)
        offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        neighbours = np.empty(int(offsets[-1]), dtype=np.int32)
        scores = np.empty(int(offsets[-1]), dtype=float)
        filled = np.zeros(size, dtype=np.int64)
        for left, right, pair_scores in self._filtered_pairs(chunks_factory()):
            sources, destinations = np.concatenate((left, right)), np.concatenate((right, left))
            directed_scores = np.concatenate((pair_scores, pair_scores))
            order = np.argsort(sources, kind="stable")
            sources = sources[order]
            unique, starts, chunk_counts = np.unique(sources, return_index=True, return_counts=True)
            if np.any(filled[unique] + chunk_counts > counts[unique]):
                raise ValueError("Similarity input changed between retrieval-index passes")
            positions = np.repeat(offsets[unique] + filled[unique] - starts, chunk_counts)
            positions += np.arange(len(sources), dtype=np.int64)
            neighbours[positions] = destinations[order]
            scores[positions] = directed_scores[order]
            filled[unique] += chunk_counts
        if not np.array_equal(filled, counts):
            raise ValueError("Similarity input changed between retrieval-index passes")
        compact_offsets = np.zeros(size + 1, dtype=np.int64)
        written = 0
        for query in range(size):
            start, stop = offsets[query:query + 2]
            row_uids, row_scores = neighbours[start:stop], scores[start:stop]
            if len(row_uids):
                by_uid = np.argsort(row_uids, kind="stable")
                row_uids, row_scores = row_uids[by_uid], row_scores[by_uid]
                unique_uids, first_indices = np.unique(row_uids, return_index=True)
                minimum_scores = np.minimum.reduceat(row_scores, first_indices)
                maximum_scores = np.maximum.reduceat(row_scores, first_indices)
                if np.any(maximum_scores - minimum_scores > 1e-12):
                    raise ValueError(f"Conflicting duplicate similarity pair for query {self._uids[query]!r}")
                # Max within numerical tolerance makes duplicate collapse
                # independent of file order even for last-bit roundoff.
                row_uids, row_scores = unique_uids, maximum_scores
                rank_order = np.lexsort((row_uids, -row_scores))
                neighbours[written:written + len(row_uids)] = row_uids[rank_order]
                written += len(row_uids)
            compact_offsets[query + 1] = written
        self._offsets = compact_offsets
        self._neighbour_indices = neighbours[:written].copy()

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def query_uids(self, *, split: str = "test") -> list[str]:
        selected = self.metadata[self.metadata[self.split_col].astype(str) == str(split)]
        return sorted(selected[self.uid_col].astype(str).tolist())

    def _rng(self, query: int) -> np.random.Generator:
        payload = json.dumps(
            ["metapli-context-v1", self.policy, self.seed, self.epoch,
             str(self._targets[query]), str(self._groups[query]), str(self._splits[query]), str(self._uids[query])],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
        return np.random.default_rng(seed)

    def episode(self, query_uid: str) -> Episode:
        query_uid = str(query_uid)
        query = self._uid_to_index.get(query_uid)
        if query is None:
            raise KeyError(f"Unknown query UID: {query_uid}")
        if self.policy in ("FPSimTopK", "FPSimNorm"):
            start, stop = self._offsets[query:query + 2]
            candidates = self._neighbour_indices[start:stop]
            candidates = candidates[self._finite[candidates]]
            pool_key = (str(self._targets[query]), str(self._groups[query]), str(self._splits[query]))
            pool = self._pools.get(pool_key, np.empty(0, dtype=np.int32))
            expected = pool[pool != query]
            if len(candidates) != len(expected):
                raise ValueError(f"Similarity table is incomplete for query {query_uid!r}: "
                                 f"{len(candidates)} of {len(expected)} eligible contexts have scores")
            self._require_contexts(query_uid, len(candidates))
            if self.policy == "FPSimTopK" or self.k == 1:
                selected = candidates[:self.k]
            else:
                ranks = np.linspace(0, len(candidates) - 1, self.k, dtype=int)
                selected = candidates[ranks]
        else:
            key = (str(self._targets[query]), str(self._groups[query]), str(self._splits[query]))
            candidates = self._pools.get(key, np.empty(0, dtype=np.int32))
            candidates = candidates[candidates != query]
            self._require_contexts(query_uid, len(candidates))
            rng = self._rng(query)
            probabilities = None
            if self.policy == "ActivityBased":
                positive = candidates[self._labels[candidates] > 0]
                if len(positive) >= self.k:
                    candidates = positive
                    labels = self._labels[candidates]
                    # Algebraically identical to normalized 1/label, without
                    # overflowing for very small positive labels.
                    weights = labels.min() / labels
                    probabilities = weights / weights.sum()
            selected = rng.choice(candidates, size=self.k, replace=False, p=probabilities)
        contexts = tuple(str(self._uids[index]) for index in selected)
        if len(contexts) != self.k or query_uid in contexts or len(set(contexts)) != self.k:
            raise AssertionError("Invalid context set: wrong K, query inclusion or duplicate UID")
        return Episode(
            query_uid=query_uid, context_uids=contexts,
            query_label=float(self._labels[query]),
            context_labels=tuple(float(self._labels[index]) for index in selected),
            group_id=str(self._groups[query]),
        )

    def _require_contexts(self, query_uid: str, available: int) -> None:
        if available < self.k:
            raise ValueError(
                f"Query {query_uid!r} has {available} eligible contexts for {self.policy}; "
                f"exactly {self.k} are required"
            )
