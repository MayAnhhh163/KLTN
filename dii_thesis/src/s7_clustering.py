# s7_clustering.py
# Step 7 (UPDATED): Phân cụm quốc gia dựa trên DII-Core (2015–2022 average)
#
# Input:
#   - data/processed/dii_core/dii_core_country_avg.csv
#
# Output (data/processed/dii_core/clustering/):
#   - clusters_kmeans.csv
#   - clusters_hierarchical.csv
#   - dendrogram.png
#   - k_selection_largest_jump.csv
#   - cluster_profiles.csv
#
# Ghi chú:
# - Clustering dùng 3 trụ (pillar_*_mean). Đây là cách “giàu thông tin” hơn so với chỉ dùng DII tổng.
# - Với hierarchical Ward, k được chọn bằng "largest jump" trên linkage distances (đúng yêu cầu trước đó).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

from dii_thesis.src.s0_settings import PROCESSED_DIR


FEATURE_COLS = [
    "pillar_access_adoption_mean",
    "pillar_infra_capacity_mean",
    "pillar_human_capital_mean",
]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_country_avg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"country_iso3", "dii_core_z_mean"} | set(FEATURE_COLS)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong dii_core_country_avg.csv: {missing}")

    # Lọc những nước có đủ 3 trụ (tránh méo clustering)
    df = df.dropna(subset=FEATURE_COLS).copy()
    return df


def _zscore_matrix(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def _largest_jump_k_from_linkage(Z: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, pd.DataFrame]:
    # Z: scipy linkage matrix
    dists = Z[:, 2]
    n = Z.shape[0] + 1  # number of observations
    jumps = np.full_like(dists, np.nan, dtype=float)
    jumps[1:] = dists[1:] - dists[:-1]

    rows: List[Dict] = []
    for m in range(2, n):  # merge step (1-indexed)
        k_if_cut = n - (m - 1)
        if k_min <= k_if_cut <= k_max:
            rows.append({
                "k": int(k_if_cut),
                "step_m": int(m),
                "distance": float(dists[m - 1]),
                "jump": float(jumps[m - 1]),
            })

    df = pd.DataFrame(rows).sort_values("k", ascending=True)
    if df.empty:
        return k_min, df

    # largest jump => chọn k tương ứng với jump lớn nhất
    best = df.loc[df["jump"].idxmax()]
    return int(best["k"]), df


def choose_k_kmeans_silhouette_constrained(
    df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    min_cluster_share: float = 0.03,
    random_state: int = 42,
    n_init: int = 50,
) -> Tuple[int, pd.DataFrame, str]:
    """
    Chọn k cho KMeans bằng silhouette, có ràng buộc kích thước cụm tối thiểu.

    Quy tắc:
    - Với mỗi k trong [k_min, k_max], fit KMeans trên 3 trụ (đã z-score).
    - Loại (ineligible) các k mà cụm nhỏ nhất < min_cluster_share * N.
    - Chọn k có silhouette cao nhất trong các k hợp lệ.
    - Nếu không có k nào hợp lệ, fallback chọn k có silhouette cao nhất (và ghi policy rõ ràng).

    Output:
    - best_k: k được chọn
    - table: bảng kết quả theo k (silhouette, inertia, min_cluster_share, eligible)
    - policy: mô tả quy tắc chọn k (để ghi vào luận văn/phụ lục)
    """
    X = _zscore_matrix(df[FEATURE_COLS].to_numpy(dtype=float))
    n = len(df)
    k_hi = min(k_max, n - 1)

    rows: List[Dict] = []
    for k in range(k_min, k_hi + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)

        sizes = pd.Series(labels).value_counts()
        min_size = int(sizes.min())
        min_share = float(min_size / n)
        eligible = bool(min_share >= min_cluster_share)

        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = float("nan")

        rows.append({
            "k": int(k),
            "silhouette": sil,
            "inertia": float(model.inertia_),
            "min_cluster_size": min_size,
            "min_cluster_share": min_share,
            "eligible": eligible,
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return k_min, tbl, "fallback_empty"

    # chọn theo eligible trước
    tbl_eligible = tbl[tbl["eligible"]].copy()

    if len(tbl_eligible) > 0 and tbl_eligible["silhouette"].notna().any():
        best_row = tbl_eligible.loc[tbl_eligible["silhouette"].idxmax()]
        policy = f"silhouette_max_with_min_cluster_share>={min_cluster_share}"
        best_k = int(best_row["k"])
    else:
        # fallback: chọn k có silhouette cao nhất (không ràng buộc)
        if tbl["silhouette"].notna().any():
            best_row = tbl.loc[tbl["silhouette"].idxmax()]
            best_k = int(best_row["k"])
            policy = f"fallback_silhouette_max_no_k_meets_min_cluster_share>={min_cluster_share}"
        else:
            best_k = int(tbl["k"].min())
            policy = "fallback_min_k_all_nan_silhouette"

    return best_k, tbl, policy




def kmeans_cluster(df: pd.DataFrame, k: int, random_state: int = 42) -> pd.DataFrame:
    X = _zscore_matrix(df[FEATURE_COLS].to_numpy(dtype=float))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=50)
    labels = model.fit_predict(X)
    out = df.copy()
    out["cluster_kmeans"] = labels + 1  # 1..k
    return out


def hierarchical_ward_cluster(df: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> Tuple[pd.DataFrame, int, pd.DataFrame, np.ndarray]:
    X = _zscore_matrix(df[FEATURE_COLS].to_numpy(dtype=float))
    Z = linkage(X, method="ward", metric="euclidean")

    k_best, k_table = _largest_jump_k_from_linkage(Z, k_min=k_min, k_max=k_max)
    labels = fcluster(Z, t=k_best, criterion="maxclust")

    out = df.copy()
    out["cluster_hierarchical"] = labels
    return out, k_best, k_table, Z


def cluster_profiles(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    cols = ["dii_core_z_mean"] + FEATURE_COLS
    prof = (
        df.groupby(label_col, as_index=False)[cols]
        .mean()
        .rename(columns={label_col: "cluster"})
        .sort_values("cluster")
    )
    counts = df.groupby(label_col, as_index=False).size().rename(columns={label_col: "cluster", "size": "n_countries"})
    return prof.merge(counts, on="cluster", how="left")


def save_dendrogram(Z: np.ndarray, labels: List[str], out_png: Path) -> None:
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main(
    input_country_avg: Path,
    outdir: Path,
    kmeans_k: int = 0,
    k_min: int = 2,
    k_max: int = 10,
    min_cluster_share: float = 0.03,
) -> None:
    _ensure_dir(outdir)

    df = _read_country_avg(input_country_avg)

    # KMeans: chọn k tự động nếu kmeans_k <= 0
    k_auto, k_tbl, k_policy = choose_k_kmeans_silhouette_constrained(
        df,
        k_min=k_min,
        k_max=k_max,
        min_cluster_share=min_cluster_share,
    )
    k_tbl.to_csv(outdir / "k_selection_kmeans_silhouette.csv", index=False)

    if kmeans_k <= 0:
        kmeans_k = k_auto
        kmeans_policy = k_policy
    else:
        kmeans_policy = "user_fixed_k"

    df_km = kmeans_cluster(df, k=kmeans_k)
    df_km.to_csv(outdir / "clusters_kmeans.csv", index=False)
    cluster_profiles(df_km, "cluster_kmeans").to_csv(outdir / "cluster_profiles_kmeans.csv", index=False)


    # Hierarchical Ward (k chọn bằng largest jump)
    df_h, k_best, k_table, Z = hierarchical_ward_cluster(df, k_min=k_min, k_max=k_max)
    df_h.to_csv(outdir / "clusters_hierarchical.csv", index=False)
    k_table.to_csv(outdir / "k_selection_largest_jump.csv", index=False)
    cluster_profiles(df_h, "cluster_hierarchical").to_csv(outdir / "cluster_profiles_hierarchical.csv", index=False)

    # Dendrogram
    labels = df_h["country_iso3"].astype(str).tolist()
    save_dendrogram(Z, labels=labels, out_png=outdir / "dendrogram.png")

    print(f"Saved clustering outputs to: {outdir}")
    print(f"Hierarchical k_best={k_best} | KMeans k={kmeans_k} ({kmeans_policy}) | countries_used={len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_country_avg",
        type=str,
        default=str(PROCESSED_DIR / "dii_core" / "dii_core_country_avg.csv"),
        help="Path tới dii_core_country_avg.csv (Step 5 output).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROCESSED_DIR / "dii_core" / "clustering"),
        help="Output directory cho clustering.",
    )
    parser.add_argument("--kmeans_k", type=int, default=0, help="Số cụm KMeans. Nếu <=0 thì tự chọn k bằng silhouette (có ràng buộc cụm nhỏ nhất).")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--min_cluster_share", type=float, default=0.03, help="Ràng buộc: loại k nếu cụm nhỏ nhất < min_cluster_share * N.")
    args = parser.parse_args()

    main(
        Path(args.input_country_avg),
        Path(args.outdir),
        kmeans_k=args.kmeans_k,
        k_min=args.k_min,
        k_max=args.k_max,
        min_cluster_share=args.min_cluster_share,
    )
