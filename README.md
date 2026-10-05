# MetaPLI

MetaPLI predicts protein–ligand bioactivity from pocket–ligand structures and
labelled support complexes. An MPNN encodes structural interactions, and an ANP
produces a context-conditioned activity prediction and uncertainty.

## Installation

The code uses Python 3.9 and PyTorch 1.13.1. Dependency versions are listed in
`environment.yml`. Install DGL, torch-scatter and torch-sparse builds matching
your PyTorch and CUDA environment, then install the package:

```bash
python -m pip install --no-deps -e .
```

## Data preparation

Use the prepared activity/structure records and the corresponding Exact-target
or Homology split manifest. Data and model checkpoints will be deposited on
Zenodo. Input formats and file layout are described in
[data/README.md](data/README.md).

Run the following commands from the repository root:

```bash
metapli-prepare-data --records data/records.csv \
  --split data/exact_target_split.csv --output data/metadata.csv
metapli-build-graphs --metadata data/metadata.csv \
  --output-lmdb data/graphs.lmdb
metapli-build-similarity --metadata data/metadata.csv \
  --output data/similarities.csv
```

For the Homology protocol, use its corresponding records and
`data/homology40_split.csv`. Graph inputs are prepared pocket MOL2 and
posed-ligand SDF files. The default protein–ligand contact cutoff is 6 Å.
Fingerprint retrieval uses ECFP4 (radius 2, 1024 bits) and Tanimoto similarity.

## Context retrieval

| Method | Support selection |
|---|---|
| `FPSimTopK` | Select the K most similar eligible ligands. |
| `FPSimNorm` | Select K uniformly spaced positions in the similarity ranking; K=1 uses the nearest ligand. |
| `Random` | Uniformly sample K eligible records without replacement. |
| `ActivityBased` | Sample using inverse positive pIC50 weights; use uniform sampling when fewer than K positive labels are available. |

Support records share the query's target, structure source and split, have
measured activities, and exclude the query itself. Set `context.policy`,
`context.k` and `context.seed` in the configuration. The top-level `seed`
controls model randomness.

## Training and prediction

Update the data paths and context settings in `configs/train.yaml` and
`configs/predict.yaml`. Set the prediction checkpoint to the corresponding
trained model. YAML paths are relative to the configuration file's directory.

```bash
metapli-train --config configs/train.yaml
metapli-predict --config configs/predict.yaml
```

Training uses Adam with Gaussian negative log likelihood, ListMLE ranking and
posterior–prior KL divergence. Checkpoints are selected by validation negative
log likelihood. Prediction conditions on support activities with fixed network
weights and outputs predicted activity, standard deviation, variance and the
support record identifiers.

## Evaluation

Compute per-target RMSE, Pearson R and Spearman ρ, with equal-weight averages
across targets:

```bash
metapli-evaluate outputs/predictions.csv --metadata data/metadata.csv \
  --split test --per-target-output outputs/metrics.csv \
  --summary-output outputs/summary.json
```
