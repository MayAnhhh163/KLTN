# s7_clustering.py
# Step 7: Clustering countries by Digital Inclusion Index (multi-spec)
#
# For each spec:
# - Choose k using silhouette (k-means) or largest-jump in linkage distances (hierarchical Ward)
# - Run clustering
# - Export cluster assignments + summaries + cross-tabs
#
# Run:
#   python dii_thesis/src/s7_clustering.py \
#   --spec_root dii_thesis/data/processed/spec_outputs \
#   --spec ALL
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster


# -----------------------------
# Specs definition (aligned with Step 6)
# Note: Thesis currently uses S1..S5 (S4 removed and indices shifted)
# -----------------------------
SPECS = {
    "S1": {"method": "kmeans"},
    "S2": {"method": "kmeans"},
    "S3": {"method": "kmeans"},
    "S4": {"method": "hierarchical"},
    "S5": {"method": "hierarchical"},
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

    n = len(X)
    if n < 3:
        df = pd.DataFrame([{"k": 2, "silhouette": np.nan}])
        return {"best_k": 2, "scores": df}

    for k in range(k_min, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=20)
        labels = km.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan
        scores.append({"k": k, "silhouette": sil})

    df = pd.DataFrame(scores)

    if df["silhouette"].notna().any():
        best_row = df.loc[df["silhouette"].idxmax()]
        best_k = int(best_row["k"])
    else:
        best_k = int(df["k"].min())

    return {"best_k": best_k, "scores": df}


def run_kmeans(df: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=0, n_init=50)
    return km.fit_predict(df)


def run_hierarchical(df: np.ndarray, k: int) -> np.ndarray:
    Z = linkage(df, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels - 1  # make clusters 0-based


def choose_k_largest_jump_from_linkage(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
) -> Dict:
    """
    Choose k for hierarchical Ward using "largest jump" in linkage distances, without plotting dendrogram.

    Logic:
    - Compute Ward linkage Z on X (n x d)
    - Linkage distances are Z[:, 2], length = n-1
    - Let d[m] = distance at merge step m (1..n-1), increasing monotonically for Ward
    - Consider jumps: jump[m] = d[m] - d[m-1] (m>=2)
    - Each merge reduces number of clusters by 1:
        after merge step m, #clusters = n - m
    - A "large jump" indicates forced merge of dissimilar clusters; cut BEFORE that merge.
      If largest jump occurs at step m, chosen k = n - (m-1)

    We restrict candidate k into [k_min, k_max] to keep typology interpretable.
    Return dict with best_k, jump_table, and basic linkage stats.
    """
    n = len(X)
    if n < 3:
        # too few points; default to k=2
        return {
            "best_k": 2,
            "jump_table": pd.DataFrame([{"k": 2, "step_m": np.nan, "distance": np.nan, "jump": np.nan}]),
            "linkage_distances": [],
            "note": "n<3; default k=2",
        }

    Z = linkage(X, method="ward")
    dists = Z[:, 2].astype(float)  # length n-1

    # Compute jumps (first jump undefined)
    jumps = np.empty_like(dists)
    jumps[:] = np.nan
    jumps[1:] = dists[1:] - dists[:-1]

    # Map merge step m (1-indexed) to k if cut before that merge:
    # if largest jump at step m (m>=2), choose k = n - (m-1)
    # We'll build candidate table only for k in [k_min, k_max]
    rows: List[Dict] = []
    for m in range(2, n):  # m in 2..n-1 (1-indexed)
        k_if_cut = n - (m - 1)
        if k_min <= k_if_cut <= k_max:
            rows.append(
                {
                    "k": int(k_if_cut),
                    "step_m": int(m),
                    "distance": float(dists[m - 1]),  # m-1 index
                    "jump": float(jumps[m - 1]),
                }
            )

    if not rows:
        # If n small or constraints too tight, fall back to closest feasible k
        best_k = min(max(k_min, 2), min(k_max, n))
        jt = pd.DataFrame([{"k": best_k, "step_m": np.nan, "distance": np.nan, "jump": np.nan}])
        return {
            "best_k": int(best_k),
            "jump_table": jt,
            "linkage_distances": dists.tolist(),
            "note": "No candidate k in [k_min,k_max]; fallback used",
        }

    jt = pd.DataFrame(rows).sort_values(["k"], ascending=True)

    # Choose k corresponding to max jump
    # If tie, prefer smaller k (more interpretable) by sorting k ascending after jump desc
    jt2 = jt.sort_values(["jump", "k"], ascending=[False, True])
    best_k = int(jt2.iloc[0]["k"])

    return {
        "best_k": best_k,
        "jump_table": jt,
        "linkage_distances": dists.tolist(),
        "note": "k chosen by largest linkage-distance jump within constrained k-range",
    }


def summarize_clusters(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Cluster summary table:
    - size
    - mean DII
    - region composition (unique set)
    - income group composition (unique set)
    """
    if "region" not in df.columns:
        df["region"] = "Unknown"
    if "income_group" not in df.columns:
        df["income_group"] = "Unknown"

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


def attach_country_metadata(df: pd.DataFrame, spec_root: Path) -> pd.DataFrame:
    """
    Ensure df has region and income_group.
    If missing, merge from processed/country_dim.csv.
    Assumption: spec_root = .../data/processed/spec_outputs
    => processed_dir = spec_root.parent
    """
    need_cols = {"region", "income_group"}
    if need_cols.issubset(set(df.columns)):
        return df

    processed_dir = spec_root.parent
    country_dim_path = processed_dir / "country_dim.csv"
    if not country_dim_path.exists():
        return df

    country_dim = pd.read_csv(country_dim_path)

    cols = ["country_iso3", "country_name", "region", "income_group"]
    cols = [c for c in cols if c in country_dim.columns]
    country_dim = country_dim[cols].copy()

    df2 = df.merge(country_dim, on="country_iso3", how="left", suffixes=("", "_dim"))

    if "region" not in df.columns and "region_dim" in df2.columns:
        df2["region"] = df2["region_dim"]
    elif "region" in df.columns and "region_dim" in df2.columns:
        df2["region"] = df2["region"].fillna(df2["region_dim"])

    if "income_group" not in df.columns and "income_group_dim" in df2.columns:
        df2["income_group"] = df2["income_group_dim"]
    elif "income_group" in df.columns and "income_group_dim" in df2.columns:
        df2["income_group"] = df2["income_group"].fillna(df2["income_group_dim"])

    for c in ["region_dim", "income_group_dim"]:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])

    return df2


def save_cluster_crosstabs(df: pd.DataFrame, spec_dir: Path) -> None:
    """
    Save pd.crosstab(cluster, region/income_group) as CSV (counts + row-normalized percent).
    """
    if "cluster" not in df.columns:
        return

    if "region" in df.columns:
        ct_region = pd.crosstab(df["cluster"], df["region"])
        ct_region_pct = pd.crosstab(df["cluster"], df["region"], normalize="index")
        ct_region.to_csv(spec_dir / "cluster_by_region_counts.csv")
        ct_region_pct.to_csv(spec_dir / "cluster_by_region_pct.csv")

    if "income_group" in df.columns:
        ct_income = pd.crosstab(df["cluster"], df["income_group"])
        ct_income_pct = pd.crosstab(df["cluster"], df["income_group"], normalize="index")
        ct_income.to_csv(spec_dir / "cluster_by_income_counts.csv")
        ct_income_pct.to_csv(spec_dir / "cluster_by_income_pct.csv")


# -----------------------------
# Main per-spec runner
# -----------------------------
def run_spec(spec_root: Path, spec: str) -> None:
    spec_dir = spec_root / f"spec_{spec}"
    ensure_dir(spec_dir)

    df = load_index_scores(spec_dir)

    if "DII_primary" not in df.columns:
        raise ValueError(f"[{spec}] index_scores.csv phải có cột 'DII_primary' để clustering.")

    df = attach_country_metadata(df, spec_root)

    X = df[["DII_primary"]].to_numpy()
    method = SPECS[spec]["method"]

    report: Dict = {
        "spec": spec,
        "cluster_on": "DII_primary",
        "n_countries": int(df["country_iso3"].nunique()) if "country_iso3" in df.columns else int(len(df)),
    }

    if method == "kmeans":
        k_info = choose_k_silhouette(X, k_min=2, k_max=6)
        best_k = k_info["best_k"]
        labels = run_kmeans(X, best_k)

        report["method"] = "kmeans"
        report["best_k"] = int(best_k)
        report["k_selection_method"] = "silhouette"
        report["silhouette_table"] = k_info["scores"].to_dict(orient="records")

    else:
        # hierarchical Ward with k chosen by largest jump in linkage distances
        k_info = choose_k_largest_jump_from_linkage(X, k_min=2, k_max=6)
        best_k = k_info["best_k"]
        labels = run_hierarchical(X, best_k)

        report["method"] = "hierarchical_ward"
        report["chosen_k"] = int(best_k)
        report["k_selection_method"] = "largest_jump_linkage_distance"
        report["note"] = k_info.get("note", "")
        report["jump_table"] = k_info["jump_table"].to_dict(orient="records")

    df["cluster"] = labels

    out_assign = spec_dir / "cluster_assignments.csv"
    df.to_csv(out_assign, index=False)

    summary = summarize_clusters(df, "cluster")
    out_summary = spec_dir / "cluster_summary.csv"
    summary.to_csv(out_summary, index=False)

    save_cluster_crosstabs(df, spec_dir)

    # add cluster size distribution to report
    report["cluster_sizes"] = df["cluster"].value_counts().sort_index().to_dict()

    out_report = spec_dir / "clustering_report.json"
    import json
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[{spec}] clustering done | method={report.get('method')} | k={best_k}")
    print("  saved: cluster_assignments.csv, cluster_summary.csv, clustering_report.json")
    print("  saved: cluster_by_region_counts.csv, cluster_by_region_pct.csv")
    print("  saved: cluster_by_income_counts.csv, cluster_by_income_pct.csv")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Step 7: Clustering for DII multi-spec framework")
    ap.add_argument("--spec_root", type=str, required=True)
    ap.add_argument("--spec", type=str, default="ALL", help="S1..S5 or ALL")
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
