# s7_clustering.py
# Step 7: Clustering countries by Digital Inclusion Index (multi-spec)
#
# For each spec:
# - Choose k using silhouette (k-means) or dendrogram (hierarchical)
# - Run clustering
# - Export cluster assignments + summaries
#
# Run:
#   python dii_thesis/src/s7_clustering.py \
#   --spec_root dii_thesis/data/processed/spec_outputs \
#   --spec ALL
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster


# -----------------------------
# Specs definition (aligned with Step 6)
# -----------------------------
SPECS = {
    "S1": {"method": "kmeans"},
    "S2": {"method": "kmeans"},
    "S3": {"method": "kmeans"},
    "S4": {"method": "kmeans"},
    "S5": {"method": "hierarchical"},
    "S6": {"method": "hierarchical"},
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_index_scores(spec_dir: Path) -> pd.DataFrame:
    path = spec_dir / "index_scores.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def choose_k_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 6) -> Dict:
    """
    Choose k using silhouette score.
    Returns dict with best_k and table of scores.
    """
    scores = []
    for k in range(k_min, min(k_max, len(X)) + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores.append({"k": k, "silhouette": sil})

    df = pd.DataFrame(scores)
    best_row = df.loc[df["silhouette"].idxmax()]
    return {
        "best_k": int(best_row["k"]),
        "scores": df
    }


def run_kmeans(df: pd.DataFrame, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=0, n_init=50)
    return km.fit_predict(df)


def run_hierarchical(df: pd.DataFrame, k: int) -> np.ndarray:
    Z = linkage(df, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels - 1  # make clusters 0-based


def summarize_clusters(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Cluster summary table:
    - size
    - mean DII
    - region composition
    - income group composition
    """
    rows = []
    for c in sorted(df[label_col].unique()):
        sub = df[df[label_col] == c]
        rows.append({
            "cluster": int(c),
            "n_countries": len(sub),
            "mean_DII": sub["DII_primary"].mean(),
            "regions": ", ".join(sorted(sub["region"].dropna().unique())),
            "income_groups": ", ".join(sorted(sub["income_group"].dropna().unique()))
        })
    return pd.DataFrame(rows)


# -----------------------------
# Main per-spec runner
# -----------------------------
def run_spec(spec_root: Path, spec: str) -> None:
    spec_dir = spec_root / f"spec_{spec}"
    ensure_dir(spec_dir)

    df = load_index_scores(spec_dir)

    # clustering variable: primary DII
    X = df[["DII_primary"]].to_numpy()

    method = SPECS[spec]["method"]

    report = {}

    if method == "kmeans":
        # choose k by silhouette
        k_info = choose_k_silhouette(X, k_min=2, k_max=6)
        best_k = k_info["best_k"]

        labels = run_kmeans(X, best_k)

        report["method"] = "kmeans"
        report["best_k"] = best_k
        report["silhouette_table"] = k_info["scores"].to_dict(orient="records")

    else:
        # hierarchical Ward
        # choose k conservatively = 3 (common in global typologies)
        best_k = 3
        labels = run_hierarchical(X, best_k)

        report["method"] = "hierarchical_ward"
        report["chosen_k"] = best_k
        report["note"] = "k fixed to 3 for interpretability and comparability"

    df["cluster"] = labels

    # Save cluster assignment
    out_assign = spec_dir / "cluster_assignments.csv"
    df.to_csv(out_assign, index=False)

    # Save cluster summary
    summary = summarize_clusters(df, "cluster")
    out_summary = spec_dir / "cluster_summary.csv"
    summary.to_csv(out_summary, index=False)

    # Save report
    out_report = spec_dir / "clustering_report.json"
    import json
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[{spec}] clustering done | method={report.get('method')} | k={best_k}")
    print(f"  saved: cluster_assignments.csv, cluster_summary.csv, clustering_report.json")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Step 7: Clustering for DII multi-spec framework")
    ap.add_argument("--spec_root", type=str, required=True)
    ap.add_argument("--spec", type=str, default="ALL", help="S1..S6 or ALL")
    return ap.parse_args()


def main():
    args = parse_args()
    spec_root = Path(args.spec_root)

    if args.spec.upper() == "ALL":
        specs = list(SPECS.keys())
    else:
        specs = [args.spec.upper()]

    for s in specs:
        run_spec(spec_root, s)


if __name__ == "__main__":
    main()
