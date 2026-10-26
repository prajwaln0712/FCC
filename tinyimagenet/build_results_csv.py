"""
Build four decluttered per-(score, aggregation) CSVs instead of one single merged table, joined
by (source, target) for all 90 directed task pairs, with Pearson r / Spearman rho appended as
labeled summary rows at the bottom of each file (each FCC column's correlation against both gain
columns -- a correlation coefficient is one number across all 90 rows, not a per-pair value, so it
doesn't belong as an ordinary per-row column).

Accuracy source files (JSON .txt, written by transfer_eval.py / eval_baselineB.py):
  finetune/source_{S}/results_target_{T}.txt   -- real transfer (S's pretrained backbone -> T's fc)
  baselineB/results_task_{T}.txt               -- full-network-trained-on-T-alone (per task, reused
                                                   across all 9 sources targeting T)
  baselineA/results_task_{T}.txt               -- random-frozen-backbone baseline (per task, reused
                                                   across all 9 sources targeting T)
Uses each file's "best_acc" field by default; baselineB's file only has "acc" (single epoch-20
checkpoint evaluation, not best-ever-seen).

FCC source files (written by compute_fcc_score1.py / compute_fcc_score2.py):
  tables/fcc_score1.csv  -- Inception-192 + DINO-vits16 (no augmentation)
  tables/fcc_score2.csv  -- Inception-768 + DINO-vitb16 (augmented to ~800 samples/class)

Output (replaces the old single tables/transfer_gain_fcc.csv):
  tables/tiny_split_result_fcc1_mean.csv   (fcc_inception192_mean,  fcc_dinovits16_mean)
  tables/tiny_split_result_fcc1_ratio.csv  (fcc_inception192_ratio, fcc_dinovits16_ratio)
  tables/tiny_split_result_fcc2_mean.csv   (fcc_inception768_mean,  fcc_dinovitb16_mean)
  tables/tiny_split_result_fcc2_ratio.csv  (fcc_inception768_ratio, fcc_dinovitb16_ratio)
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

NUM_TASKS = 10

# table name -> (source fcc csv, column 1, column 2)
TABLES = {
    "fcc1_mean": ("fcc_score1.csv", "fcc_inception192_mean", "fcc_dinovits16_mean"),
    "fcc1_ratio": ("fcc_score1.csv", "fcc_inception192_ratio", "fcc_dinovits16_ratio"),
    "fcc2_mean": ("fcc_score2.csv", "fcc_inception768_mean", "fcc_dinovitb16_mean"),
    "fcc2_ratio": ("fcc_score2.csv", "fcc_inception768_ratio", "fcc_dinovitb16_ratio"),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_fcc_csv(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[(int(row["source"]), int(row["target"]))] = row
    return rows


def r(x, ndigits):
    """round() that passes through None (a missing FCC lookup) unchanged."""
    return None if x is None else round(float(x), ndigits)


def main():
    parser = argparse.ArgumentParser(description="Build the 4 per-score/aggregation gain+FCC CSVs")
    parser.add_argument("--acc-field", default="best_acc", choices=["best_acc", "final_acc"],
                        help="Which accuracy field to use from finetune/ and baselineA/ results files")
    args = parser.parse_args()

    os.makedirs("tables", exist_ok=True)

    # Replaced by the 4 per-score/aggregation tables below -- remove the stale single-file merge
    # if an earlier run of this script (or the old version of it) left one behind.
    old_path = "tables/transfer_gain_fcc.csv"
    if os.path.exists(old_path):
        os.remove(old_path)
        print(f"Removed stale {old_path}")

    baselineB_acc = {t: load_json(f"baselineB/results_task_{t}.txt")["acc"] for t in range(NUM_TASKS)}
    baselineA_acc = {t: load_json(f"baselineA/results_task_{t}.txt")[args.acc_field] for t in range(NUM_TASKS)}

    base_rows = []
    for source in range(NUM_TASKS):
        for target in range(NUM_TASKS):
            if source == target:
                continue
            transfer = load_json(f"finetune/source_{source}/results_target_{target}.txt")
            transfer_acc = transfer[args.acc_field]
            b_acc = baselineB_acc[target]
            a_acc = baselineA_acc[target]
            base_rows.append({
                "source": source,
                "target": target,
                "transfer_acc": r(transfer_acc, 2),
                "baselineB_acc": r(b_acc, 2),
                "baselineA_acc": r(a_acc, 2),
                "gain_vs_baselineB": r(transfer_acc - b_acc, 2),
                "gain_vs_baselineA": r(transfer_acc - a_acc, 2),
            })

    for table_name, (fcc_file, col1, col2) in TABLES.items():
        fcc = load_fcc_csv(f"tables/{fcc_file}")
        rows = []
        for base in base_rows:
            f = fcc.get((base["source"], base["target"]), {})
            row = dict(base)
            row[col1] = r(f.get(col1), 4)
            row[col2] = r(f.get(col2), 4)
            rows.append(row)

        fieldnames = list(rows[0].keys())

        gain_b = np.array([row["gain_vs_baselineB"] for row in rows], dtype=float)
        gain_a = np.array([row["gain_vs_baselineA"] for row in rows], dtype=float)
        summary_rows = []
        for stat_name, stat_fn in [("pearson_r", pearsonr), ("spearman_rho", spearmanr)]:
            for gain_name, gain_vals in [("baselineB", gain_b), ("baselineA", gain_a)]:
                summary = {k: "" for k in fieldnames}
                summary["source"] = f"{stat_name}_vs_{gain_name}"
                for col in (col1, col2):
                    vals = np.array([row[col] for row in rows], dtype=float)
                    stat_value, _p = stat_fn(vals, gain_vals)
                    summary[col] = round(float(stat_value), 4)
                summary_rows.append(summary)

        out_path = f"tables/tiny_split_result_{table_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerows(summary_rows)
        print(f"Saved {out_path} ({len(rows)} data rows + {len(summary_rows)} summary rows)")


if __name__ == "__main__":
    main()
