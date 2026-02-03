# s9_robustness_checks.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from dii_thesis.src.s0_settings import PROCESSED_DIR

CORE_INDICATORS = [
    "IT.NET.USER.ZS",
    "IT.CEL.SETS.P2",
    "IT.NET.BBND.P2",
    "IT.NET.SECR.P6",
    "SE.SEC.ENRR",
    "SE.TER.ENRR",
]

PILLARS = {
    "pillar_access": ["IT.NET.USER.ZS", "IT.CEL.SETS.P2"],
    "pillar_infra":  ["IT.NET.BBND.P2", "IT.NET.SECR.P6"],
    "pillar_humcap": ["SE.SEC.ENRR", "SE.TER.ENRR"],
}


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")


def _safe_minmax(series: pd.Series) -> pd.Series:
    mn = series.min(skipna=True)
    mx = series.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.nan, index=series.index)
    return (series - mn) / (mx - mn)


def compute_dii_from_z(
    df: pd.DataFrame,
    strict_pillar: bool,
    min_pillars_for_dii: int = 2,
) -> pd.DataFrame:
    """
    Tính DII dựa trên *_z đã có sẵn trong dii_core_panel.csv.
    - strict_pillar=False: trụ tính nếu có >=1/2 biến (như baseline)
    - strict_pillar=True: trụ chỉ tính nếu đủ 2/2 biến
    - DII tính nếu có >= min_pillars_for_dii trụ
    """
    out = df.copy()

    # đảm bảo zcols
    zcols = [f"{c}_z" for c in CORE_INDICATORS]
    _require_cols(out, zcols)

    pillar_cols = []
    for pillar, codes in PILLARS.items():
        cols = [f"{c}_z" for c in codes]
        nonmiss = out[cols].notna().sum(axis=1)
        out[pillar] = out[cols].mean(axis=1, skipna=True)

        if strict_pillar:
            out.loc[nonmiss < len(cols), pillar] = np.nan
        else:
            out.loc[nonmiss < 1, pillar] = np.nan

        pillar_cols.append(pillar)

    out["n_pillars_available"] = out[pillar_cols].notna().sum(axis=1)
    out["dii_z"] = out[pillar_cols].mean(axis=1, skipna=True)
    out.loc[out["n_pillars_available"] < min_pillars_for_dii, "dii_z"] = np.nan

    # scale 0-100 theo p01-p99 của chính phiên bản này
    valid = out["dii_z"].dropna()
    if len(valid) > 0:
        p01, p99 = valid.quantile([0.01, 0.99])
        clipped = out["dii_z"].clip(lower=p01, upper=p99)
        out["dii_0_100"] = 100 * (clipped - p01) / (p99 - p01)
    else:
        out["dii_0_100"] = np.nan

    return out


def compute_dii_minmax(
    df: pd.DataFrame,
    strict_pillar: bool,
    min_pillars_for_dii: int = 2,
) -> pd.DataFrame:
    """
    Tính DII bằng min-max (0-1) trên toàn sample pooled 2015–2022.
    Dùng trực tiếp các cột indicator trong panel (đã transform log1p cho secure servers).
    """
    out = df.copy()
    _require_cols(out, CORE_INDICATORS)

    # minmax pooled
    for c in CORE_INDICATORS:
        out[f"{c}_mm"] = _safe_minmax(out[c])

    pillar_cols = []
    for pillar, codes in PILLARS.items():
        cols = [f"{c}_mm" for c in codes]
        nonmiss = out[cols].notna().sum(axis=1)
        out[pillar] = out[cols].mean(axis=1, skipna=True)

        if strict_pillar:
            out.loc[nonmiss < len(cols), pillar] = np.nan
        else:
            out.loc[nonmiss < 1, pillar] = np.nan

        pillar_cols.append(pillar)

    out["n_pillars_available"] = out[pillar_cols].notna().sum(axis=1)
    out["dii_mm"] = out[pillar_cols].mean(axis=1, skipna=True)
    out.loc[out["n_pillars_available"] < min_pillars_for_dii, "dii_mm"] = np.nan

    # scale 0-100 theo min-max của dii_mm (bám logic minmax)
    valid = out["dii_mm"].dropna()
    if len(valid) > 0:
        mn, mx = valid.min(), valid.max()
        if mx > mn:
            out["dii_mm_0_100"] = 100 * (out["dii_mm"] - mn) / (mx - mn)
        else:
            out["dii_mm_0_100"] = np.nan
    else:
        out["dii_mm_0_100"] = np.nan

    return out


def country_period_avg(df: pd.DataFrame, value_col: str, start: int, end: int) -> pd.DataFrame:
    d = df[(df["year"] >= start) & (df["year"] <= end)].copy()
    g = (
        d.groupby(["country_iso3", "country_name"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: f"{value_col}_avg_{start}_{end}"})
    )
    return g


def add_rank(df: pd.DataFrame, value_col: str, ascending: bool = False, rank_col: str = "rank") -> pd.DataFrame:
    out = df.copy()
    # dense rank, không nhảy số khi tie
    out[rank_col] = out[value_col].rank(ascending=ascending, method="dense")
    return out


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    # pandas corr(method='spearman') tự xử lý rank
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 3:
        return float("nan")
    return float(tmp["a"].corr(tmp["b"], method="spearman"))


def main(panel_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    _require_cols(df, ["country_iso3", "country_name", "year", "dii_core_z", "dii_core_0_100"])

    # =========================
    # Check 1: 2015–2019 vs 2020–2022 (baseline)
    # =========================
    pre = country_period_avg(df, "dii_core_0_100", 2015, 2019)
    post = country_period_avg(df, "dii_core_0_100", 2020, 2022)

    c1 = pre.merge(post, on=["country_iso3", "country_name"], how="outer")
    c1["delta_post_minus_pre"] = c1["dii_core_0_100_avg_2020_2022"] - c1["dii_core_0_100_avg_2015_2019"]

    c1 = add_rank(c1, "dii_core_0_100_avg_2015_2019", ascending=False, rank_col="rank_2015_2019")
    c1 = add_rank(c1, "dii_core_0_100_avg_2020_2022", ascending=False, rank_col="rank_2020_2022")
    c1["rank_change_post_minus_pre"] = c1["rank_2020_2022"] - c1["rank_2015_2019"]

    c1.to_csv(outdir / "robustness_check1_pre_vs_post_country.csv", index=False)

    # summary check1
    summary1 = {
        "n_countries_with_pre": int(pre["country_iso3"].nunique()),
        "n_countries_with_post": int(post["country_iso3"].nunique()),
        "spearman_rank_corr_pre_post": spearman_corr(c1["rank_2015_2019"], c1["rank_2020_2022"]),
        "median_abs_rank_change": float(np.nanmedian(np.abs(c1["rank_change_post_minus_pre"]))),
        "share_abs_rank_change_gt_10": float(np.nanmean(np.abs(c1["rank_change_post_minus_pre"]) > 10)),
    }

    # =========================
    # Check 2: Min-max vs baseline (country average 2015–2022)
    # =========================
    df_mm = compute_dii_minmax(df, strict_pillar=False, min_pillars_for_dii=2)
    mm_avg = (
        df_mm.groupby(["country_iso3", "country_name"], as_index=False)["dii_mm_0_100"]
        .mean()
        .rename(columns={"dii_mm_0_100": "dii_minmax_0_100_mean_2015_2022"})
    )

    base_avg = (
        df.groupby(["country_iso3", "country_name"], as_index=False)["dii_core_0_100"]
        .mean()
        .rename(columns={"dii_core_0_100": "dii_baseline_0_100_mean_2015_2022"})
    )

    c2 = base_avg.merge(mm_avg, on=["country_iso3", "country_name"], how="inner")
    c2 = add_rank(c2, "dii_baseline_0_100_mean_2015_2022", ascending=False, rank_col="rank_baseline")
    c2 = add_rank(c2, "dii_minmax_0_100_mean_2015_2022", ascending=False, rank_col="rank_minmax")
    c2["rank_change_minmax_minus_baseline"] = c2["rank_minmax"] - c2["rank_baseline"]

    c2.to_csv(outdir / "robustness_check2_minmax_vs_baseline_country.csv", index=False)

    summary2 = {
        "n_countries_compared": int(len(c2)),
        "spearman_rank_corr_baseline_vs_minmax": spearman_corr(c2["rank_baseline"], c2["rank_minmax"]),
        "median_abs_rank_change": float(np.nanmedian(np.abs(c2["rank_change_minmax_minus_baseline"]))),
        "share_abs_rank_change_gt_10": float(np.nanmean(np.abs(c2["rank_change_minmax_minus_baseline"]) > 10)),
    }

    # =========================
    # Check 3: Strict missing (pillar requires 2/2)
    # =========================
    # (a) strict pillar, DII requires >=2 pillars (giữ rule cũ để so sánh)
    df_strict = compute_dii_from_z(df, strict_pillar=True, min_pillars_for_dii=2)
    strict_avg = (
        df_strict.groupby(["country_iso3", "country_name"], as_index=False)["dii_0_100"]
        .mean()
        .rename(columns={"dii_0_100": "dii_strict_0_100_mean_2015_2022"})
    )

    c3 = base_avg.merge(strict_avg, on=["country_iso3", "country_name"], how="inner")
    c3 = add_rank(c3, "dii_baseline_0_100_mean_2015_2022", ascending=False, rank_col="rank_baseline")
    c3 = add_rank(c3, "dii_strict_0_100_mean_2015_2022", ascending=False, rank_col="rank_strict")
    c3["rank_change_strict_minus_baseline"] = c3["rank_strict"] - c3["rank_baseline"]
    c3.to_csv(outdir / "robustness_check3_strictmissing_vs_baseline_country.csv", index=False)

    # coverage impact
    # tỷ lệ quan sát country-year có dii tính được
    base_obs = float(df["dii_core_0_100"].notna().mean())
    strict_obs = float(df_strict["dii_0_100"].notna().mean())

    summary3 = {
        "n_countries_compared": int(len(c3)),
        "spearman_rank_corr_baseline_vs_strict": spearman_corr(c3["rank_baseline"], c3["rank_strict"]),
        "median_abs_rank_change": float(np.nanmedian(np.abs(c3["rank_change_strict_minus_baseline"]))),
        "share_abs_rank_change_gt_10": float(np.nanmean(np.abs(c3["rank_change_strict_minus_baseline"]) > 10)),
        "share_observations_with_dii_baseline": base_obs,
        "share_observations_with_dii_strict": strict_obs,
    }

    # =========================
    # Ghi summary tổng hợp
    # =========================
    summary = {
        "panel_path": str(panel_path),
        "check1_pre_vs_post": summary1,
        "check2_minmax_vs_baseline": summary2,
        "check3_strictmissing_vs_baseline": summary3,
    }

    with open(outdir / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # thêm 1 bảng tổng hợp nhanh cho luận văn
    df_sum = pd.DataFrame([
        {"check": "C1_pre_vs_post", **summary1},
        {"check": "C2_minmax_vs_baseline", **summary2},
        {"check": "C3_strictmissing_vs_baseline", **summary3},
    ])
    df_sum.to_csv(outdir / "robustness_summary_table.csv", index=False)

    print("Done. Outputs written to:", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_panel = (PROCESSED_DIR / "dii_core" / "dii_core_panel.csv")
    default_outdir = (PROCESSED_DIR / "dii_core" / "robustness")

    parser.add_argument(
        "--panel",
        type=str,
        default=str(default_panel),
        help="Path to dii_core_panel.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(default_outdir),
        help="Output directory for robustness tables",
    )
    args = parser.parse_args()

    panel_path = Path(args.panel)
    outdir = Path(args.outdir)

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file panel: {panel_path}\n"
            f"Gợi ý: chạy từ thư mục root của repo, hoặc truyền --panel đúng path."
        )

    main(panel_path, outdir)
