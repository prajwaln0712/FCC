import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
OUT_DIR = os.path.join(HERE, "..", "tables", "augmentation")


def load_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "results*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def write_table(results, metric, out_name):
    table = [
        {
            "seed_no": r["seed"],
            "inception": r["fcc"]["inception"][metric],
            "dino": r["fcc"]["dino"][metric],
            "acc_adapt": r["cifar10_acc_adapt"],
        }
        for r in results
    ]
    table.sort(key=lambda row: row["acc_adapt"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed_no", "inception", "dino", "acc_adapt"])
        w.writeheader()
        w.writerows(table)
    print(f"Wrote {out_path} ({len(table)} rows)")


if __name__ == "__main__":
    results = load_results()
    write_table(results, "mean", "aug_mean.csv")
    write_table(results, "ratio", "aug_ratio.csv")
