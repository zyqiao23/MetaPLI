# Software validation

Date: 2026-10-05.

## Verified environment

Linux, Python 3.9.18, PyTorch 1.13.1, DGL 1.0.0+cu117,
DGL-LifeSci 0.3.2, torch-scatter 2.1.2, torch-sparse 0.6.18, RDKit 2023.09.5,
ProLIF 2.0.3 and pytest 8.0.2. Tests ran on CPU with CUDA devices hidden.

## Result

**95 tests passed**. Two dependency deprecation warnings came from ProLIF's
use of `pkg_resources`.

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
  python -m pytest -q
```

The checks cover:

- Strict checkpoint loading, both supported parameter-key layouts, model
  predictions, training gradients, posterior invariance to query padding,
  rejection of entirely padded tasks and the 180° angle-bin boundary.
- The four manuscript retrieval policies, row-order/seed reproducibility,
  target/source/split eligibility, self exclusion, missing context handling
  and complete eligible fingerprint-pair coverage.
- One-to-one archived split/record merging, target-disjoint partitions,
  portable structure paths and retention of all input records.
- Synthetic prepared MOL2/SDF structures through graph construction, LMDB
  loading and batching; failed rows make graph construction fail as a whole.
- A real small-model optimizer run including a final singleton ranking group,
  complete train/validation coverage, frozen full-split prediction and basic
  per-target evaluation.
- Unmeasured queries, invalid input rejection, unknown configuration rejection,
  complete evaluation-split matching and preservation of undefined correlations.

During model migration, the parameter names were checked against the reference
implementation: all 136 parameter keys were retained. With identical weights,
valid synthetic inputs and paired random states, forward outputs and training
gradients agreed exactly before the padding correction. Release preparation
added the explicit torch-scatter dependency and corrected latent pooling to
average only unmasked positions for variable-length query batches. This
correction retains all parameter keys and shapes; the command-line workflow
uses one query per task and has no query padding. The angle-bin boundary fix
also preserves the parameter layout and bounds the index for a collinear 180°
angle.

An additional 14 independent synthetic checks passed after these corrections.
They cover query-label independence under all four context policies,
test-label independence of training and checkpoint selection, failed or
non-finite predictions, and complete evaluation input matching.

All six command wrappers passed `--help`, Python sources passed parsing, and the
package wheel built successfully. Scope inspection covered source, scripts and
configuration files. The original code directory's 89 files were unchanged.

These are software checks using synthetic inputs. Data, model checkpoints and
the manuscript's scientific results are supplied through the separate archive.
