# s9b_robustness_jrc.py
# Robustness bổ sung theo chuẩn OECD/JRC:
# (5.4) weighting sensitivity, (5.5) leave-one-out indicators, (5.6) aggregation function
#
# Input:
#   data/processed/dii_core/dii_core_panel.csv  (Step 5 output)
#
# Output (data/processed/dii_core/robustness_jrc/):
#   - robustness_weights_pca_vs_equal_country.csv
#   - robustness_leave_one_out_summary.csv
#   - robustness_leave_one_out_country.csv
#   - robustness_aggregation_geom_vs_arith_country.csv
#   - robustness_jrc_summary.json
#
# Run:
#   python dii_thesis/src/s9b_robustness_jrc.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dii_thesis.src.s0_settings import (
    PROCESSED_DIR,
    STUDY_START_YEAR,
    STUDY_END_YEAR,
    DII_CORE_INDICATORS,
    DII_PILLARS,
    MIN_INDICATORS_PER_PILLAR,
    MIN_PILLARS_FOR_DII,
)

# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _require_cols(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{where}] Thiếu cột bắt buộc: {missing}")

def _spearman(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 10:
        return float("nan")
    return float(tmp["a"].corr(tmp["b"], method="spearman"))

def _dense_rank_desc(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="dense")

def _median_abs_rank_change(r1: pd.Series, r2: pd.Series) -> float:
    tmp = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
    if len(tmp) == 0:
        return float("nan")
    return float((tmp["r1"] - tmp["r2"]).abs().median())

def _share_abs_rank_gt(r1: pd.Series, r2: pd.Series, thr: int = 10) -> float:
    tmp = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
    if len(tmp) == 0:
        return float("nan")
    return float(((tmp["r1"] - tmp["r2"]).abs() > thr).mean())

def _read_panel(panel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(panel_path)
    _require_cols(df, ["country_iso3", "year"], where="panel")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["country_iso3", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= STUDY_START_YEAR) & (df["year"] <= STUDY_END_YEAR)].copy()
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df = df[df["country_iso3"].str.len() == 3].copy()

    # đảm bảo z-cols tồn tại (Step 5 đã tạo)
    zcols = [f"{c}_z" for c in DII_CORE_INDICATORS]
    _require_cols(df, zcols, where="panel")
    return df

def _compute_pillars_from_z(df: pd.DataFrame, *, pillars: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Tính pillar score từ *_z có sẵn theo đúng rule luận văn:
    - pillar = mean z trong trụ
    - chỉ hợp lệ khi đủ MIN_INDICATORS_PER_PILLAR (mặc định 2/2)
    """
    out = df.copy()
    for pillar, inds in pillars.items():
        zcols = [f"{i}_z" for i in inds]
        n_obs = out[zcols].notna().sum(axis=1)
        out[pillar] = out[zcols].mean(axis=1, skipna=True)
        out.loc[n_obs < MIN_INDICATORS_PER_PILLAR, pillar] = np.nan
    return out

def _compute_dii_weighted_z(
    df: pd.DataFrame,
    *,
    pillar_cols: List[str],
    weights: Dict[str, float],
    min_pillars_for_dii: int = MIN_PILLARS_FOR_DII,
) -> pd.Series:
    """
    Tính DII_z có trọng số trên các pillar có sẵn (skipna),
    nhưng chỉ hợp lệ khi có >= min_pillars_for_dii pillar.
    Trọng số được chuẩn hóa lại trên các pillar khả dụng từng quan sát.
    """
    w = pd.Series({k: float(v) for k, v in weights.items()})
    w = w.reindex(pillar_cols)

    X = df[pillar_cols].copy()
    avail = X.notna()

    # chuẩn hoá trọng số theo từng hàng dựa trên các trụ available
    w_mat = np.tile(w.values, (len(df), 1))
    w_mat = np.where(avail.values, w_mat, np.nan)
    w_sum = np.nansum(w_mat, axis=1)

    # weighted mean (avoid divide-by-zero warnings)
    num = np.nansum(X.values * w_mat, axis=1)

    out_vals = np.full(len(df), np.nan, dtype=float)
    valid = np.isfinite(w_sum) & (w_sum > 0)
    out_vals[valid] = num[valid] / w_sum[valid]

    out = pd.Series(out_vals, index=df.index)

    n_pillars_avail = avail.sum(axis=1)
    out.loc[n_pillars_avail < min_pillars_for_dii] = np.nan
    return out

def _country_mean_rank(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Lấy trung bình 2015–2022 theo quốc gia, sau đó xếp hạng (dense rank desc).
    """
    meta_cols = [c for c in ["country_name", "region", "income_group", "lending_type"] if c in df.columns]
    gcols = ["country_iso3"] + meta_cols

    tmp = (
        df.groupby(gcols, as_index=False)
        .agg(
            score_mean=(score_col, "mean"),
            years_with_score=(score_col, lambda s: int(s.notna().sum())),
        )
    )
    tmp["rank"] = _dense_rank_desc(tmp["score_mean"])
    return tmp

# -----------------------------
# (5.4) Weighting sensitivity: PCA weights on pillars
# -----------------------------

def _pca_weights_from_country_pillars(country_df: pd.DataFrame, pillar_cols: List[str]) -> Dict[str, float]:
    """
    PCA trên ma trận (country x pillars) dùng điểm trụ trung bình 2015–2022.
    Lấy PC1 loading (abs) làm proxy weights, sau đó chuẩn hoá tổng = 1.
    (Thiết kế này đủ để sensitivity, không nhằm "tối ưu" chỉ số.)
    """
    X = country_df[pillar_cols].copy()
    X = X.dropna(axis=0)  # PCA cần đủ 3 trụ
    if len(X) < 10:
        raise ValueError("Không đủ quốc gia có đủ 3 trụ để ước lượng PCA weights.")

    # center
    Xc = X - X.mean(axis=0)

    # SVD PCA
    U, S, Vt = np.linalg.svd(Xc.values, full_matrices=False)
    loadings = Vt[0, :]  # PC1
    w = np.abs(loadings)
    w = w / w.sum()
    return {pillar_cols[i]: float(w[i]) for i in range(len(pillar_cols))}

def run_check_weights_pca(panel: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, dict]:
    pillar_cols = list(DII_PILLARS.keys())

    df = _compute_pillars_from_z(panel, pillars=DII_PILLARS)
    # baseline equal weights
    w_equal = {p: 1.0 / len(pillar_cols) for p in pillar_cols}
    df["dii_equal_z"] = _compute_dii_weighted_z(df, pillar_cols=pillar_cols, weights=w_equal)

    # country means for PCA estimation (need full pillars)
    ctry_pillars = _country_mean_rank(df, score_col=pillar_cols[0])  # dummy to get meta columns
    # rebuild properly: country mean pillars
    meta_cols = [c for c in ["country_name", "region", "income_group", "lending_type"] if c in df.columns]
    gcols = ["country_iso3"] + meta_cols
    country_p = df.groupby(gcols, as_index=False).agg(**{f"{p}_mean": (p, "mean") for p in pillar_cols})
    # create PCA matrix with pillar means
    country_pillars = country_p[[f"{p}_mean" for p in pillar_cols]].rename(
        columns={f"{p}_mean": p for p in pillar_cols}
    )
    weights_pca = _pca_weights_from_country_pillars(country_pillars, pillar_cols=pillar_cols)

    df["dii_pca_z"] = _compute_dii_weighted_z(df, pillar_cols=pillar_cols, weights=weights_pca)

    # ranks on country mean
    base = _country_mean_rank(df, "dii_equal_z").rename(columns={"score_mean": "dii_equal_z_mean", "rank": "rank_equal"})
    alt = _country_mean_rank(df, "dii_pca_z").rename(columns={"score_mean": "dii_pca_z_mean", "rank": "rank_pca"})

    merged = base.merge(alt[["country_iso3", "dii_pca_z_mean", "rank_pca"]], on="country_iso3", how="inner")
    merged["delta_rank"] = merged["rank_pca"] - merged["rank_equal"]

    # summary
    summ = {
        "n_countries_compared": int(len(merged)),
        "spearman_rank_corr_equal_vs_pca": _spearman(merged["rank_equal"], merged["rank_pca"]),
        "median_abs_rank_change": _median_abs_rank_change(merged["rank_equal"], merged["rank_pca"]),
        "share_abs_rank_change_gt_10": _share_abs_rank_gt(merged["rank_equal"], merged["rank_pca"], thr=10),
        "weights_pca": weights_pca,
    }

    merged.sort_values("rank_equal").to_csv(outdir / "robustness_weights_pca_vs_equal_country.csv", index=False)
    return merged, summ

# -----------------------------
# (5.5) Leave-one-out indicator set
# -----------------------------

def _pillars_from_indicator_subset(indicators: List[str]) -> Dict[str, List[str]]:
    """
    Với DII_PILLARS cố định 2 biến/trụ, khi leave-one-out, một trụ có thể còn 1 biến.
    Để vẫn 'đúng tinh thần' robustness: giữ cấu trúc trụ, nhưng pillar chỉ hợp lệ khi đủ 2/2
    => trụ có biến bị loại sẽ gần như luôn NA, phản ánh tính nhạy thật của cấu trúc.
    """
    # vẫn giữ mapping gốc, nhưng compute_pillars_from_z sẽ yêu cầu 2/2 theo MIN_INDICATORS_PER_PILLAR
    return DII_PILLARS

def run_check_leave_one_out(panel: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pillar_cols = list(DII_PILLARS.keys())

    df0 = _compute_pillars_from_z(panel, pillars=DII_PILLARS)
    # baseline equal weights
    w_equal = {p: 1.0 / len(pillar_cols) for p in pillar_cols}
    df0["dii_base_z"] = _compute_dii_weighted_z(df0, pillar_cols=pillar_cols, weights=w_equal)

    base_rank = _country_mean_rank(df0, "dii_base_z")[["country_iso3", "rank"]].rename(columns={"rank": "rank_baseline"})

    rows_summary: List[dict] = []
    rows_country: List[pd.DataFrame] = []

    for removed in DII_CORE_INDICATORS:
        df = panel.copy()

        # "remove" indicator by setting its z column to NaN
        zcol = f"{removed}_z"
        if zcol not in df.columns:
            continue
        df[zcol] = np.nan

        df = _compute_pillars_from_z(df, pillars=_pillars_from_indicator_subset(DII_CORE_INDICATORS))
        df["dii_alt_z"] = _compute_dii_weighted_z(df, pillar_cols=pillar_cols, weights=w_equal)

        alt_rank = _country_mean_rank(df, "dii_alt_z")[["country_iso3", "rank"]].rename(columns={"rank": "rank_alt"})
        m = base_rank.merge(alt_rank, on="country_iso3", how="inner")
        m["delta_rank"] = m["rank_alt"] - m["rank_baseline"]
        m["indicator_removed"] = removed
        rows_country.append(m)

        rows_summary.append({
            "indicator_removed": removed,
            "n_countries_compared": int(len(m)),
            "spearman_rank_corr": _spearman(m["rank_baseline"], m["rank_alt"]),
            "median_abs_rank_change": _median_abs_rank_change(m["rank_baseline"], m["rank_alt"]),
            "share_abs_rank_change_gt_10": _share_abs_rank_gt(m["rank_baseline"], m["rank_alt"], thr=10),
        })

    country_all = pd.concat(rows_country, ignore_index=True) if rows_country else pd.DataFrame()
    summary = pd.DataFrame(rows_summary).sort_values("median_abs_rank_change", ascending=False)

    country_all.to_csv(outdir / "robustness_leave_one_out_country.csv", index=False)
    summary.to_csv(outdir / "robustness_leave_one_out_summary.csv", index=False)

    summ_json = {
        "n_indicators": int(len(DII_CORE_INDICATORS)),
        "max_median_abs_rank_change": float(summary["median_abs_rank_change"].max()) if len(summary) else float("nan"),
        "min_spearman_rank_corr": float(summary["spearman_rank_corr"].min()) if len(summary) else float("nan"),
    }
    return country_all, summary, summ_json

# -----------------------------
# (5.6) Aggregation: arithmetic vs geometric mean
# -----------------------------

def _geo_mean_rowwise(X: pd.DataFrame) -> pd.Series:
    """
    Geometric mean cho các giá trị có thể âm:
    Dùng signed-log transform: g(x)=sign(x)*log1p(|x|), lấy mean rồi invert bằng sinh.
    Đây là cách 'bảo thủ' để có phiên bản ít bù trừ hơn trên thang z, mà vẫn xử lý số âm.
    """
    A = X.copy()
    # signed log1p
    G = np.sign(A) * np.log1p(np.abs(A))
    m = G.mean(axis=1, skipna=True)
    out = np.sign(m) * (np.expm1(np.abs(m)))
    return out

def run_check_aggregation(panel: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, dict]:
    pillar_cols = list(DII_PILLARS.keys())
    df = _compute_pillars_from_z(panel, pillars=DII_PILLARS)

    # arithmetic baseline (equal weights)
    w_equal = {p: 1.0 / len(pillar_cols) for p in pillar_cols}
    df["dii_arith_z"] = _compute_dii_weighted_z(df, pillar_cols=pillar_cols, weights=w_equal)

    # geometric-like alternative (less compensable)
    # Step 1: "geo" at pillar level? pillars already means of 2 indicators.
    # Here we apply geometric-like aggregation at index level across pillars.
    Xp = df[pillar_cols].copy()
    df["dii_geom_z"] = _geo_mean_rowwise(Xp)
    # enforce missingness: need >=2 pillars like baseline
    df.loc[Xp.notna().sum(axis=1) < MIN_PILLARS_FOR_DII, "dii_geom_z"] = np.nan

    base = _country_mean_rank(df, "dii_arith_z").rename(columns={"rank": "rank_arithmetic", "score_mean": "dii_arith_z_mean"})
    alt = _country_mean_rank(df, "dii_geom_z").rename(columns={"rank": "rank_geometric", "score_mean": "dii_geom_z_mean"})

    merged = base.merge(alt[["country_iso3", "dii_geom_z_mean", "rank_geometric"]], on="country_iso3", how="inner")
    merged["delta_rank"] = merged["rank_geometric"] - merged["rank_arithmetic"]

    summ = {
        "n_countries_compared": int(len(merged)),
        "spearman_rank_corr_arith_vs_geom": _spearman(merged["rank_arithmetic"], merged["rank_geometric"]),
        "median_abs_rank_change": _median_abs_rank_change(merged["rank_arithmetic"], merged["rank_geometric"]),
        "share_abs_rank_change_gt_10": _share_abs_rank_gt(merged["rank_arithmetic"], merged["rank_geometric"], thr=10),
    }

    merged.sort_values("rank_arithmetic").to_csv(outdir / "robustness_aggregation_geom_vs_arith_country.csv", index=False)
    return merged, summ

# -----------------------------
# Main
# -----------------------------

def main(panel_path: Path, outdir: Path) -> None:
    _ensure_dir(outdir)
    panel = _read_panel(panel_path)

    weights_country, summ_w = run_check_weights_pca(panel, outdir)
    loo_country, loo_summary, summ_loo = run_check_leave_one_out(panel, outdir)
    agg_country, summ_agg = run_check_aggregation(panel, outdir)

    summary = {
        "panel_path": str(panel_path),
        "check54_weights_pca": summ_w,
        "check55_leave_one_out": summ_loo,
        "check56_aggregation_geom": summ_agg,
    }
    (outdir / "robustness_jrc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Done. Outputs saved to:", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--panel",
        type=str,
        default=str(PROCESSED_DIR / "dii_core" / "dii_core_panel.csv"),
        help="Path tới dii_core_panel.csv (Step 5 output).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(PROCESSED_DIR / "dii_core" / "robustness_jrc"),
        help="Output directory.",
    )
    args = p.parse_args()
    main(Path(args.panel), Path(args.outdir))
