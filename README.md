# FCC (Fréchet Class Coverage): experiments

Code for testing whether FCC predicts transfer-learning performance. FCC compares a source dataset,
represented by one Gaussian (mean and covariance of its embeddings), against each class of a target
dataset (one Gaussian per class), using the Fréchet distance. Two aggregates are reported: `mean`
(average of the per-class distances) and `ratio` (mean divided by max). The figures in this repo use
the **ratio** version.

Experiments, in the order of this README:

1. **NIST**: 5 digit/letter datasets, 20 directed pairs.
2. **Augmentation** (`dataaugmentation/`): Tiny-ImageNet → CIFAR-10, one point per augmentation seed.
3. **ImageNet-1K** (`imagenet1k/`): ImageNet-1K → 7 target datasets.
4. **LogME and LEEP** (`other_metrics/`, `LogME/`): comparison metrics for experiments 2 and 3.

Then: **Generating the plots**.

## Repository layout

| Path | Contents |
|---|---|
| `nist/` | NIST experiment |
| `dataaugmentation/` | Augmentation experiment |
| `imagenet1k/` | ImageNet-1K experiment |
| `LogME/` | LogME and LEEP implementations (You et al.) |
| `other_metrics/` | LogME/LEEP scripts and correlation tables |
| `models/`, `data.py`, `util.py` | Shared models, dataset loaders, Fréchet-distance code |
| `plots.py` | Figure code (PDF and SVG output) |
| `jobs/` | SLURM scripts, one folder per experiment |
| `tables/`, `plots/` | Result tables and figures |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Run every command from the repository root.
- DINOv3 models are gated on Hugging Face: accept the licence and run `huggingface-cli login` first.
- Most datasets download automatically into `./data`. Place these three by hand:
  - Tiny-ImageNet-200 at `data/tiny-ImageNet/tiny-imagenet-200`
  - CIFAR-10 at `data/CIFAR10` (augmentation experiment)
  - ImageNet-1K at `/shared/rc/nlagent/data/imagenet` (or pass `--imagenet-root`)
- The scripts in `jobs/` are SLURM scripts for the RIT cluster. Each one runs the matching command below.

---

## 1. NIST experiment

```bash
python -m nist.verify_mnist_datasets
python -m nist.nist_extract_embeddings --network both
python -m nist.nist_experiment --features inception
python -m nist.nist_experiment --features dino
```

The `nist_distances_cache_*.npz` files are written to the current directory. Move them into `nist/`,
next to `nist_pg_vals.json`.

---

## 2. Augmentation experiment

```bash
python -m dataaugmentation.augment --seed 1
```

Run once per seed (the results use seeds 1–9, 13, 14, 15). Each run writes
`dataaugmentation/aug_<seed>/results.json`. Copy each one to `dataaugmentation/results/results<seed>.json`,
then build the tables:

```bash
python dataaugmentation/build_aug_tables.py
```

---

## 3. ImageNet-1K experiment

Targets: `cifar10 places365 eurosat svhn mnist fashionmnist kmnist`.

All stages on SLURM:

```bash
bash jobs/imagenet1k/submit_all_imagenet1k.sh
```

Or one stage at a time (repeat the two `--target` commands for every target):

```bash
python -m imagenet1k.verify_targets
python -m imagenet1k.smoke_test_dino_vith16plus
python -m imagenet1k.imagenet_source_stats
python -m imagenet1k.imagenet1k_test --target cifar10
python -m imagenet1k.target_fcc_stats --target cifar10
python -m imagenet1k.fcc_experiment
python -m imagenet1k.imagenet1k_analysis
```

The last step writes `imagenet1k/tables/imagenet1k_results.csv`. Copy it to
`tables/imagenet1k/inception2048/imagenet1k_results.csv`. Design decisions are in
`Design_choices_imagenet1k.txt`.

---

## 4. LogME and LEEP (augmentation and ImageNet-1K)

Needs `numba` (in `requirements.txt`) and the trained models from experiments 2 and 3. No GPU is required.

```bash
python -m other_metrics.aug_scores
python -m other_metrics.imagenet1k_scores
python -m other_metrics.correlation_tables
```

- Add `--limit 2000` to the first two commands for a quick test (nothing is saved).
- On SLURM: `sbatch jobs/other_metrics/aug_scores.sh` and `sbatch jobs/other_metrics/imagenet1k_scores.sh`.
- Scores are written to `tables/other_metrics/{aug,imagenet1k}_scores.csv`, and the correlation tables to
  `tables/other_metrics/{aug,imagenet1k}_correlation_table.{csv,md}`.

---

## Generating the plots

Run these after the steps above. Each command writes a `.pdf` and a `.svg`.

```bash
python -c "import plots; plots.plot_nist_embedding('inception'); plots.plot_nist_embedding('dino')"
python -c "import plots; plots.plot_all_aug()"
python -c "import plots; plots.plot_all_imagenet1k()"
```

| Experiment | Output folder | Reads |
|---|---|---|
| NIST | `plots/nist/` | `nist/nist_distances_cache_{inception,dino}.npz`, `nist/nist_pg_vals.json` |
| Augmentation | `plots/augment/` | `tables/augmentation/aug_ratio.csv` (made by `build_aug_tables.py`) |
| ImageNet-1K | `plots/imagenet1k/` | `tables/imagenet1k/inception2048/imagenet1k_results.csv` |

Fonts, sizes and colours are set at the top of `plots.py`.

---

## Known issues

- **Cluster-specific paths:** `~/ghn3_env`, `~/fcc`, `/shared/rc/nlagent/data/imagenet`, the partition
  names and the log directory `~/logs/fcc` in `jobs/` are for the RIT cluster. Adjust them on another system.
