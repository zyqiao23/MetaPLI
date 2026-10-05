# Data inputs

The data archive supplies prepared records, protocol-specific split assignments,
structure files and model checkpoints. Place the downloaded inputs under this
directory or update the configuration paths.

`records.csv` contains:

| Column | Meaning |
|---|---|
| `uid` | Unique activity/structure record identifier; also the graph LMDB key. |
| `Uniprot` | Canonical target identifier. |
| `pdb_id` | Protein structure source grouping identifier. |
| `-logAffi` | Normalized pIC50; required for training and measured support records, optional for an unmeasured prediction query. |
| `Molecule SMILES` | Ligand SMILES used to construct the ECFP4 fingerprint. |
| `pocket_path` | Path to the prepared protein pocket MOL2. |
| `ligand_path` | Path to the posed ligand SDF. |

Structure paths in the records are relative to `records.csv`, or may be explicit
local paths. All UIDs are unique and all identifier/structure fields are present.
The prepared activities retain the archive's assay-unit normalization.

Each split manifest contains exactly `uid,Split` and one row per record.
`Split` is `train`, `validation` or `test`.

- Exact-target manifests keep all records for one UniProt target in one partition.
- Homology manifests keep each sequence-defined homology component in one partition.

Use the records and split manifest from the same archived protocol/version.
`metapli-prepare-data` merges them one-to-one and retains every record. The
resulting `metadata.csv` contains the record columns plus `Split`.

Graph construction uses the actual `pocket_path` and `ligand_path` for each UID.
An archived graph LMDB can be supplied directly if its keys match the metadata.
Similarity construction writes every pair eligible by target, source and split.

Suggested local filenames, also used by the example configurations:

```text
data/
  records.csv
  exact_target_split.csv
  homology40_split.csv
  metadata.csv
  graphs.lmdb/
  similarities.csv
  model.ckpt
```

Data and checkpoint files are distributed through the archive; this repository
contains the code and file-format documentation. The publication DOI and the
archive's own file manifest identify the final downloadable version.

