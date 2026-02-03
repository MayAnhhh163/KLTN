# s5_prepare_base.py
# Step 5 (UPDATED): Xây dựng DII-Core (WDI-only) theo chuẩn composite index cho luận văn
#
# Input:
#   - data/processed/dii_panel_wide.csv  (từ Step 3)
#
# Output (data/processed/dii_core/):
#   - dii_core_panel.csv               : panel (country-year) gồm điểm trụ + DII-Core
#   - dii_core_country_avg.csv         : trung bình 2015–2022 theo quốc gia (phục vụ xếp hạng / clustering)
#   - dii_core_scalers.csv             : mean/std pooled cho từng chỉ báo (tái lập)
#   - dii_core_missing_report.csv      : coverage theo năm & theo chỉ báo
#   - dii_core_pipeline_report.json    : log tham số & thống kê mẫu
#
# Logic chính (đã thống nhất):
#   1) Chỉ dùng bộ 6 chỉ báo DII-Core
#   2) Biến đổi: log(1+x) cho IT.NET.SECR.P6
#   3) Chuẩn hóa: pooled z-score gộp toàn giai đoạn 2015–2022
#   4) Điểm trụ: trung bình z-score trong trụ, tính nếu có 2/2 chỉ báo
#   5) DII-Core: trung bình 3 trụ, tính nếu có >=2/3 trụ
#
# Run:
#   python dii_thesis/src/s5_prepare_base.py
#   python dii_thesis/src/s5_prepare_base.py --input_panel <path> --outdir <path>

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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

@dataclass(frozen=True)
class ZScaler:
    mean: float
    std: float


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"country_iso3", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong panel: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["country_iso3", "year"]).copy()
    df["year"] = df["year"].astype(int)

    # lọc giai đoạn nghiên cứu
    df = df[(df["year"] >= STUDY_START_YEAR) & (df["year"] <= STUDY_END_YEAR)].copy()

    # đảm bảo ISO3 hợp lệ (defensive)
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df = df[df["country_iso3"].str.len() == 3].copy()

    # ép numeric cho các chỉ báo core
    for c in DII_CORE_INDICATORS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _transform_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # log(1+x) cho secure servers để giảm lệch và ảnh hưởng outlier
    out["IT.NET.SECR.P6"] = np.log1p(out["IT.NET.SECR.P6"])
    return out


def _fit_pooled_scalers(df: pd.DataFrame) -> Dict[str, ZScaler]:
    scalers: Dict[str, ZScaler] = {}
    for ind in DII_CORE_INDICATORS:
        s = df[ind]
        mu = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True, ddof=0))  # pooled, ddof=0 để ổn định hơn
        if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Không thể fit z-score cho {ind} (mean/std không hợp lệ).")
        scalers[ind] = ZScaler(mean=mu, std=sd)
    return scalers


def _apply_zscore(df: pd.DataFrame, scalers: Dict[str, ZScaler]) -> pd.DataFrame:
    out = df.copy()
    for ind, sc in scalers.items():
        out[f"{ind}_z"] = (out[ind] - sc.mean) / sc.std
    return out


def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Pillar scores
    for pillar, inds in DII_PILLARS.items():
        zcols = [f"{i}_z" for i in inds]
        # số biến quan sát được trong trụ
        n_obs = out[zcols].notna().sum(axis=1)
        out[pillar] = out[zcols].mean(axis=1, skipna=True)
        out.loc[n_obs < MIN_INDICATORS_PER_PILLAR, pillar] = np.nan

    pillar_cols = list(DII_PILLARS.keys())
    out["n_pillars_available"] = out[pillar_cols].notna().sum(axis=1)

    # DII core (z-scale)
    out["dii_core_z"] = out[pillar_cols].mean(axis=1, skipna=True)
    out.loc[out["n_pillars_available"] < MIN_PILLARS_FOR_DII, "dii_core_z"] = np.nan

    # Scale 0-100 để trình bày (dùng p01-p99 pooled để bớt outlier)
    valid = out["dii_core_z"].dropna()
    if len(valid) > 0:
        p01, p99 = valid.quantile([0.01, 0.99])
        clipped = out["dii_core_z"].clip(lower=p01, upper=p99)
        out["dii_core_0_100"] = 100.0 * (clipped - p01) / (p99 - p01) if (p99 - p01) > 0 else np.nan
    else:
        out["dii_core_0_100"] = np.nan

    return out


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    years = list(range(STUDY_START_YEAR, STUDY_END_YEAR + 1))

    # tổng số quốc gia trong từng năm (sau khi lọc ISO3)
    for y in years:
        dy = df[df["year"] == y]
        n_total = int(dy["country_iso3"].nunique())
        for ind in DII_CORE_INDICATORS:
            n_nonmiss = int(dy.loc[dy[ind].notna(), "country_iso3"].nunique())
            rows.append({
                "year": y,
                "indicator_code": ind,
                "n_countries_total": n_total,
                "n_countries_non_missing": n_nonmiss,
                "coverage_rate": (n_nonmiss / n_total) if n_total else np.nan,
            })
    return pd.DataFrame(rows)


def _pipeline_report(df_scored: pd.DataFrame, scalers: Dict[str, ZScaler], outdir: Path) -> dict:
    pillar_cols = list(DII_PILLARS.keys())
    report = {
        "study_years": [STUDY_START_YEAR, STUDY_END_YEAR],
        "core_indicators": DII_CORE_INDICATORS,
        "pillars": DII_PILLARS,
        "missing_rules": {
            "min_indicators_per_pillar": MIN_INDICATORS_PER_PILLAR,
            "min_pillars_for_dii": MIN_PILLARS_FOR_DII,
        },
        "transforms": {
            "IT.NET.SECR.P6": "log1p",
        },
        "pooled_zscalers": {k: asdict(v) for k, v in scalers.items()},
        "panel_stats": {
            "n_rows": int(len(df_scored)),
            "n_countries": int(df_scored["country_iso3"].nunique()),
            "n_years": int(df_scored["year"].nunique()),
            "share_with_dii": float(df_scored["dii_core_z"].notna().mean()),
            "share_with_3_pillars": float((df_scored["n_pillars_available"] == len(pillar_cols)).mean()),
        },
    }
    # thêm thống kê theo năm (tỷ lệ quan sát có DII)
    by_year = (
        df_scored.assign(has_dii=df_scored["dii_core_z"].notna())
        .groupby("year", as_index=False)
        .agg(
            n_rows=("country_iso3", "size"),
            n_countries=("country_iso3", "nunique"),
            share_has_dii=("has_dii", "mean"),
            mean_dii=("dii_core_z", "mean"),
            median_dii=("dii_core_z", "median"),
        )
    )
    report["year_stats"] = by_year.to_dict(orient="records")
    return report


# -----------------------------
# Main
# -----------------------------

def main(input_panel: Path, outdir: Path) -> None:
    _ensure_dir(outdir)

    panel = _read_panel(input_panel)

    # Keep metadata columns if available
    meta_cols = [c for c in ["country_name", "region", "income_group", "lending_type"] if c in panel.columns]

    panel_t = _transform_indicators(panel)
    scalers = _fit_pooled_scalers(panel_t)
    panel_z = _apply_zscore(panel_t, scalers)
    scored = _compute_scores(panel_z)

    # Output panel
    zcols = [f"{i}_z" for i in DII_CORE_INDICATORS]
    out_cols = ["country_iso3"] + meta_cols + ["year"] + DII_CORE_INDICATORS + zcols + list(DII_PILLARS.keys()) + [
        "n_pillars_available", "dii_core_z", "dii_core_0_100"
    ]
    scored[out_cols].sort_values(["country_iso3", "year"]).to_csv(outdir / "dii_core_panel.csv", index=False)

    # Country average (2015-2022): dùng cho xếp hạng + clustering
    agg_cols = ["dii_core_z", "dii_core_0_100"] + list(DII_PILLARS.keys())
    country_avg = (
        scored.groupby(["country_iso3"] + meta_cols, as_index=False)
        .agg(
            **{f"{c}_mean": (c, "mean") for c in agg_cols},
            years_with_dii=("dii_core_z", lambda s: int(s.notna().sum())),
        )
        .sort_values("dii_core_z_mean", ascending=False)
    )
    country_avg.to_csv(outdir / "dii_core_country_avg.csv", index=False)

    # Scalers
    pd.DataFrame([{"indicator_code": k, "mean": v.mean, "std": v.std} for k, v in scalers.items()]).to_csv(
        outdir / "dii_core_scalers.csv", index=False
    )

    # Missing report (trước biến đổi z-score, dựa trên panel đã lọc & transform)
    _missing_report(panel_t).to_csv(outdir / "dii_core_missing_report.csv", index=False)

    # Pipeline report
    report = _pipeline_report(scored, scalers, outdir)
    (outdir / "dii_core_pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved outputs to: {outdir}")
    print(f"Panel rows={len(scored)} | countries={scored['country_iso3'].nunique()} | years={scored['year'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_panel",
        type=str,
        default=str(PROCESSED_DIR / "dii_panel_wide.csv"),
        help="Path tới file dii_panel_wide.csv (Step 3 output).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROCESSED_DIR / "dii_core"),
        help="Thư mục output cho DII-Core.",
    )
    args = parser.parse_args()
    main(Path(args.input_panel), Path(args.outdir))
