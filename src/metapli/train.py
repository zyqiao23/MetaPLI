"""Train MetaPLI on all training queries and select by validation loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .context_policies import PolicyContextIndex
from .data import EpisodeCollator, LMDBGraphStore
from .model import MetaPLI, default_model_config
from .predict import _context_config, _load_config, _resolve, _seed, _validate_keys


class TrainingModule(nn.Module):
    """Gaussian NLL, ListMLE and latent KL used by the model training objective."""

    def __init__(self, model: MetaPLI, train_config: dict) -> None:
        super().__init__()
        self.model = model
        self.nll_weight = float(train_config.get("nll_weight", 0.5))
        self.ranking_weight = float(train_config.get("listmle_weight", 0.5))
        self.clip_norm = float(train_config.get("gradient_clip_val", 3.0))
        self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction="none")

    @staticmethod
    def _listmle(prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prediction = prediction.reshape(-1)[(~mask).reshape(-1)]
        truth = truth.reshape(-1)[(~mask).reshape(-1)]
        if prediction.numel() < 2:
            return prediction.sum() * 0.0
        ordered = prediction[torch.argsort(truth, descending=True, stable=True)]
        suffix = torch.flip(torch.logcumsumexp(torch.flip(ordered, [0]), 0), [0])
        return (suffix - ordered).mean()

    @classmethod
    def _detached_ranking_gradient(cls, prediction, truth, mask):
        leaf = prediction.detach().requires_grad_(True)
        ranking = cls._listmle(leaf, truth, mask)
        return ranking.detach(), torch.autograd.grad(ranking, leaf)[0].detach()

    def _components(self, batch):
        query_graph, context_graph, truth, context_y, task_ids = batch[:5]
        if not torch.isfinite(truth).all():
            raise ValueError("Training and validation require finite query labels")
        output = self.model(query_graph, context_graph, truth if self.model.training else None, context_y, task_ids)
        mu, sigma, posterior, prior, mask = output[1], output[2], output[3], output[4], output[8]
        mu, sigma = mu.squeeze(-1), sigma.squeeze(-1)
        # Runtime collates one query per task; the model packs sorted task IDs.
        dense_truth = truth.reshape(-1, 1)
        if mu.shape != dense_truth.shape or mask.shape != dense_truth.shape:
            raise ValueError("Training requires exactly one query per task")
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("Non-finite or nonpositive predictive distribution")
        nll = self.gaussian_nll(mu, dense_truth, sigma.square().clamp_min(1e-6)).mean()
        kl = torch.distributions.kl_divergence(posterior, prior).mean() if posterior is not None else mu.new_tensor(0.0)
        if not torch.isfinite(nll) or not torch.isfinite(kl):
            raise ValueError("Non-finite training objective")
        return nll, kl, mu, dense_truth, mask

    def optimizer_step(self, batches: list[tuple], optimizer, scheduler=None) -> dict:
        """One ranking-group update with exact two-pass gradient accumulation.

        The first pass obtains the ListMLE gradient with respect to every query
        mean. The second replays the same random states and combines that gradient
        with NLL and KL, keeping one graph activation tape in memory at a time.
        A final shorter group is processed, including a singleton with zero
        ranking contribution, so no training queries are dropped.
        """
        if not batches:
            raise ValueError("Cannot train an empty ranking group")
        self.model.train()
        states, means, truths, masks, nlls, kls = [], [], [], [], [], []
        with torch.no_grad():
            for batch in batches:
                states.append((torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
                nll, kl, mu, truth, mask = self._components(batch)
                means.append(mu)
                truths.append(truth)
                masks.append(mask)
                nlls.append(nll)
                kls.append(kl)
        ranking, rank_gradient = self._detached_ranking_gradient(torch.cat(means), torch.cat(truths), torch.cat(masks))
        optimizer.zero_grad(set_to_none=True)
        cursor = 0
        for batch, (cpu_state, cuda_state), reference_mu in zip(batches, states, means):
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            nll, kl, mu, _, _ = self._components(batch)
            count = reference_mu.shape[0]
            surrogate = (mu * rank_gradient[cursor:cursor + count]).sum()
            cursor += count
            loss = (self.nll_weight * nll + kl) / len(batches) + self.ranking_weight * surrogate
            if not torch.isfinite(loss):
                raise ValueError("Non-finite optimizer objective")
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        nll, kl = torch.stack(nlls).mean(), torch.stack(kls).mean()
        return {
            "loss": float(self.nll_weight * nll + self.ranking_weight * ranking + kl),
            "nll": float(nll), "listmle": float(ranking), "kl": float(kl),
            "gradient_norm": float(norm), "n_queries": sum(mean.shape[0] for mean in means),
        }


def _validate_config(config: dict) -> tuple[dict, dict]:
    allowed = {"metadata_csv", "similarity_csv", "graph_lmdb", "output_dir", "init_checkpoint", "device", "seed", "context", "model", "training"}
    required = {"metadata_csv", "graph_lmdb", "output_dir", "device", "seed", "context", "training"}
    _validate_keys(config, allowed, required, "training config")
    context = _context_config(config["context"])
    settings = config["training"]
    _validate_keys(settings, {"epochs", "rank_size", "learning_rate", "warmup_steps", "decay", "nll_weight", "listmle_weight", "gradient_clip_val"}, {"epochs", "rank_size"}, "training")
    for key in ("epochs", "rank_size"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if float(settings.get("listmle_weight", 0.5)) > 0 and settings["rank_size"] < 2:
        raise ValueError("Positive ListMLE weight requires rank_size >= 2")
    for key, default in (("learning_rate", 1e-4), ("gradient_clip_val", 3.0)):
        value = float(settings.get(key, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    for key in ("nll_weight", "listmle_weight"):
        value = float(settings.get(key, 0.5))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be nonnegative and finite")
    if float(settings.get("nll_weight", 0.5)) + float(settings.get("listmle_weight", 0.5)) <= 0:
        raise ValueError("At least one data objective must have positive weight")
    warmup = settings.get("warmup_steps", 10000)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("training.warmup_steps must be a nonnegative integer")
    decay = float(settings.get("decay", 0.99))
    if not np.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("training.decay must be in (0, 1]")
    model_settings = config.get("model", {})
    _validate_keys(model_settings, {"pocket_encoder_config", "anp_config", "readout_scope"}, set(), "model")
    for key, keys in (
        ("pocket_encoder_config", {"hidden_dim", "radius", "T", "p_dropout", "num_heads", "in_dim_atom", "in_dim_edge"}),
        ("anp_config", {"dim_in", "dim_hid", "dim_latent", "depth", "heads", "dim_label", "dim_out"}),
    ):
        if key in model_settings:
            _validate_keys(model_settings[key], keys, set(), f"model.{key}")
    if config.get("init_checkpoint") is not None and model_settings:
        raise ValueError("Use checkpoint model configuration when init_checkpoint is provided")
    return context, dict(settings)


def _move_batch(batch, device):
    return tuple(value.to(device) if position < 5 else value for position, value in enumerate(batch))


def run(config_path: Path) -> Path:
    config_path = Path(config_path)
    config = _load_config(config_path)
    context_config, settings = _validate_config(config)
    _seed(config["seed"])
    device = torch.device(config["device"])
    metadata_path = _resolve(config_path, config["metadata_csv"])
    graph_path = _resolve(config_path, config["graph_lmdb"])
    similarity_path = _resolve(config_path, config["similarity_csv"]) if config.get("similarity_csv") is not None else None
    output = _resolve(config_path, config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output directory must be empty: {output}")
    train_index = PolicyContextIndex.from_csv(metadata_path, similarity_path, **context_config)
    val_index = PolicyContextIndex.from_csv(metadata_path, similarity_path, **context_config)
    train_uids = train_index.query_uids(split="train")
    val_uids = val_index.query_uids(split="validation")
    if not train_uids or not val_uids:
        raise ValueError("Training requires nonempty train and validation splits")
    for index, uids in ((train_index, train_uids), (val_index, val_uids)):
        labels = index.rows.loc[uids, index.label_col].to_numpy(dtype=float)
        if not np.isfinite(labels).all():
            raise ValueError("All training and validation labels must be finite")
        for uid in uids:
            index.episode(uid)
    if config.get("init_checkpoint") is not None:
        core = MetaPLI.from_checkpoint(_resolve(config_path, config["init_checkpoint"]), context_num=context_config["k"])
    else:
        model_config = default_model_config()
        for key, value in config.get("model", {}).items():
            if isinstance(value, dict):
                model_config[key].update(value)
            else:
                model_config[key] = value
        model_config["context_num"] = context_config["k"]
        core = MetaPLI(model_config)
    core = core.to(device)
    module = TrainingModule(core, settings)
    optimizer = Adam(core.parameters(), lr=float(settings.get("learning_rate", 1e-4)))
    warmup = int(settings.get("warmup_steps", 10000))
    decay = float(settings.get("decay", 0.99))

    def schedule(step):
        effective = step + 1
        return effective / warmup if effective < warmup else decay ** max(0, effective - warmup)

    scheduler = LambdaLR(optimizer, schedule)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    best_loss = float("inf")
    history = []
    with LMDBGraphStore(graph_path) as store, (output / "contexts.jsonl").open("w", encoding="utf-8") as context_log:
        collator = EpisodeCollator(store)

        def get_batch(index, uid, stage, epoch):
            episode = index.episode(uid)
            context_log.write(json.dumps({"stage": stage, "epoch": epoch, "query_uid": uid, "context_uids": episode.context_uids, "policy": context_config["policy"], "k": context_config["k"], "context_seed": context_config["seed"]}) + "\n")
            return _move_batch(collator([episode]), device)

        for epoch in range(settings["epochs"]):
            train_index.set_epoch(epoch)
            order = torch.randperm(len(train_uids)).tolist()
            reports = []
            batches = []
            for position in order:
                batches.append(get_batch(train_index, train_uids[position], "train", epoch))
                if len(batches) == settings["rank_size"]:
                    reports.append(module.optimizer_step(batches, optimizer, scheduler))
                    batches = []
            if batches:
                reports.append(module.optimizer_step(batches, optimizer, scheduler))
            observed = sum(item["n_queries"] for item in reports)
            if observed != len(train_uids):
                raise AssertionError("Every training query must be processed each epoch")
            core.eval()
            validation_nlls = []
            with torch.no_grad():
                for uid in val_uids:
                    nll, _, _, _, _ = module._components(get_batch(val_index, uid, "validation", epoch))
                    validation_nlls.append(float(nll))
            val_nll = float(np.mean(validation_nlls))
            val_loss = module.nll_weight * val_nll
            report = {
                "epoch": epoch, "train_loss": float(np.average([item["loss"] for item in reports], weights=[item["n_queries"] for item in reports])),
                "validation_nll": val_nll, "validation_loss": val_loss,
                "train_queries": observed, "validation_queries": len(validation_nlls),
                "optimizer_steps": len(reports), "last_rank_group_size": reports[-1]["n_queries"],
                "singleton_rank_groups": sum(item["n_queries"] == 1 for item in reports),
            }
            checkpoint = {"state_dict": core.state_dict(), "model_config": core.config, "epoch": epoch, "validation_nll": val_nll, "seed": config["seed"], "context_seed": context_config["seed"]}
            torch.save(checkpoint, output / "last.ckpt")
            # Validation NLL alone selects the checkpoint, even for an optional
            # zero NLL objective weight. No test measurements enter selection.
            if val_nll < best_loss:
                best_loss = val_nll
                torch.save(checkpoint, output / "best.ckpt")
            history.append(report)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(json.dumps(report, allow_nan=False), flush=True)
    (output / "training_complete.json").write_text(json.dumps({"epochs": settings["epochs"], "train_queries": len(train_uids), "validation_queries": len(val_uids), "best_validation_nll": best_loss, "checkpoint_selection": "minimum validation NLL", "seed": config["seed"], "context_seed": context_config["seed"]}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"training_output={run(args.config)}")


if __name__ == "__main__":
    main()
