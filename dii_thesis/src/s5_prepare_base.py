# s5_prepare_base.py
# Multi-spec base dataset builder for Global Digital Inclusion Index (DII)
#
# Outputs per spec:
# - base_raw.csv         : country-level averages (window) after transforms (log1p/winsorize)
# - base_zscore.csv      : z-scored indicators for PCA/clustering
# - base_report.json     : summary (countries kept/dropped, missingness, parameters)
# - coverage_by_indicator.csv : coverage rates within the window
# - inclusion_table.csv  : which countries included + reasons
#
# Run:
#    python dii_thesis/src/s5_prepare_base.py \
#   --input_panel dii_thesis/data/processed/dii_panel_wide.csv \
#   --catalog dii_thesis/config/indicator_catalog.csv \
#   --outdir dii_thesis/data/processed/spec_outputs \
#   --spec ALL
#
# Notes:
# - Deterministic (no randomness).
# - Designed for research defensibility: explicit inclusion rules & audit outputs.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Specifications (6 specs)
# -----------------------------

@dataclass(frozen=True)
class Spec:
    name: str
    time_window: Tuple[int, int]          # inclusive
    indicator_set: str                    # "core" or "core_plus_tertiary"
    index_method: str                     # for later steps; kept here for reporting
    clustering_method: str                # for later steps; kept here for reporting
    pca_level: str = "global"             # "global" or "pillar"
    notes: str = ""


SPECS: Dict[str, Spec] = {
    # S1 Main
    "S1": Spec(
        name="S1",
        time_window=(2018, 2022),
        indicator_set="core",
        index_method="pca",
        clustering_method="kmeans",
        pca_level="global",
        notes="Main spec: 2018–2022 avg, core indicators, global PCA."
    ),
    # S2 Baseline
    "S2": Spec(
        name="S2",
        time_window=(2018, 2022),
        indicator_set="core",
        index_method="equal_weight",
        clustering_method="kmeans",
        pca_level="global",
        notes="Baseline index: equal-weight (used later), same base dataset as S1."
    ),
    # S3 Temporal robustness
    "S3": Spec(
        name="S3",
        time_window=(2015, 2019),
        indicator_set="core",
        index_method="pca",
        clustering_method="kmeans",
        pca_level="global",
        notes="Temporal robustness: 2015–2019 avg, core indicators."
    ),
    # S4 Indicator sensitivity (add tertiary)
    "S4": Spec(
        name="S4",
        time_window=(2018, 2022),
        indicator_set="core_plus_tertiary",
        index_method="pca",
        clustering_method="kmeans",
        pca_level="global",
        notes="Indicator sensitivity: core + tertiary enrollment."
    ),
    # S5 Clustering robustness (hierarchical)
    "S5": Spec(
        name="S5",
        time_window=(2018, 2022),
        indicator_set="core",
        index_method="pca",
        clustering_method="hierarchical_ward",
        pca_level="global",
        notes="Clustering robustness: hierarchical Ward (used later), same base dataset as S1."
    ),
    # S6 Structural test (pillar PCA + hierarchical)
    "S6": Spec(
        name="S6",
        time_window=(2018, 2022),
        indicator_set="core",
        index_method="pca",
        clustering_method="hierarchical_ward",
        pca_level="pillar",
        notes="Structure test: pillar-level PCA (used later) + hierarchical Ward."
    ),
}


# -----------------------------
# Core defaults (defensible rules)
# -----------------------------
DEFAULT_MIN_YEARS_NON_MISSING = 3   # within a 5-year window
DEFAULT_WINSOR_P = 0.01             # winsorize at 1% and 99%
DEFAULT_ZSCORE_EPS = 1e-12


# -----------------------------
# Utility functions
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def winsorize_series(s: pd.Series, p: float) -> pd.Series:
    """Winsorize numeric series at p and 1-p quantiles."""
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
    """Z-score; if std ~ 0 -> return 0 for non-missing to avoid division errors."""
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd < DEFAULT_ZSCORE_EPS:
        return s.apply(lambda x: 0.0 if pd.notna(x) else np.nan)
    return (s - mu) / sd


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    cat = pd.read_csv(catalog_path)
    required = {"indicator_code", "pillar", "direction", "preferred_transform", "priority"}
    missing_cols = required - set(cat.columns)
    if missing_cols:
        raise ValueError(f"indicator_catalog.csv thiếu cột: {sorted(missing_cols)}")

    cat["indicator_code"] = cat["indicator_code"].astype(str)
    cat["pillar"] = cat["pillar"].astype(str)
    cat["direction"] = cat["direction"].astype(str)
    cat["preferred_transform"] = cat["preferred_transform"].fillna("none").astype(str)
    cat["priority"] = pd.to_numeric(cat["priority"], errors="coerce").fillna(999).astype(int)

    return cat


def select_indicators(cat: pd.DataFrame, indicator_set: str) -> List[str]:
    # core: priority == 1
    core = cat.loc[cat["priority"] == 1, "indicator_code"].tolist()

    if indicator_set == "core":
        return core

    if indicator_set == "core_plus_tertiary":
        # include any optional that is present (priority >=2) but specifically keep tertiary if available
        optional = cat.loc[cat["priority"] >= 2, "indicator_code"].tolist()
        # in your catalog, SE.TER.ENRR is the optional; keep it if present
        return core + [c for c in optional if c not in core]

    raise ValueError(f"indicator_set không hợp lệ: {indicator_set}")


def build_window_average(
    panel: pd.DataFrame,
    indicators: List[str],
    year0: int,
    year1: int,
    min_years_non_missing: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute country-level averages over [year0, year1] with inclusion flags per indicator:
    - For each country & indicator: require at least min_years_non_missing non-null within window, else set NaN.
    Returns:
    - avg_df: country-level, one row per country_iso3 (averaged values)
    - non_missing_counts: country-level counts per indicator within window
    """
    win = panel[(panel["year"] >= year0) & (panel["year"] <= year1)].copy()

    # Keep only necessary cols to reduce memory
    keep_cols = ["country_iso3", "country_name", "region", "income_group", "lending_type", "year"] + indicators
    keep_cols = [c for c in keep_cols if c in win.columns]
    win = win[keep_cols]

    # counts of non-missing per indicator, per country
    count_df = (
        win.groupby("country_iso3")[indicators]
        .apply(lambda g: g.notna().sum())
        .reset_index()
    )

    # mean per indicator per country
    mean_df = (
        win.groupby("country_iso3")[indicators]
        .mean(numeric_only=True)
        .reset_index()
    )

    # bring metadata (country_name/region/income/lending) from most frequent non-null
    meta_cols = [c for c in ["country_name", "region", "income_group", "lending_type"] if c in win.columns]
    if meta_cols:
        meta = (
            win.groupby("country_iso3")[meta_cols]
            .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else s.dropna().iloc[0] if not s.dropna().empty else np.nan)
            .reset_index()
        )
        avg_df = meta.merge(mean_df, on="country_iso3", how="right")
    else:
        avg_df = mean_df.copy()

    # apply min_years_non_missing rule
    counts_indexed = count_df.set_index("country_iso3")
    avg_indexed = avg_df.set_index("country_iso3")
    for ind in indicators:
        if ind not in avg_indexed.columns:
            continue
        eligible = counts_indexed[ind] >= min_years_non_missing
        # if not eligible -> NaN
        avg_indexed.loc[~eligible, ind] = np.nan

    avg_df = avg_indexed.reset_index()
    return avg_df, count_df


def apply_transforms(
    df: pd.DataFrame,
    cat: pd.DataFrame,
    indicators: List[str],
    winsor_p: float
) -> pd.DataFrame:
    out = df.copy()

    # winsorize first (robust to outliers), then transform as configured
    for ind in indicators:
        if ind not in out.columns:
            continue
        out[ind] = pd.to_numeric(out[ind], errors="coerce")
        out[ind] = winsorize_series(out[ind], winsor_p)

    # preferred_transform from catalog
    tf_map = cat.set_index("indicator_code")["preferred_transform"].to_dict()

    for ind in indicators:
        if ind not in out.columns:
            continue
        tf = tf_map.get(ind, "none")
        if tf == "log1p":
            # log1p requires non-negative; if any negative appears (rare), set to NaN (defensive)
            x = out[ind]
            out[ind] = np.where(x.notna() & (x >= 0), np.log1p(x), np.nan)
        elif tf == "none":
            pass
        else:
            # keep conservative: unknown transform => do nothing but keep note in report
            pass

    return out


def make_inclusion_rule(indicators: List[str], indicator_set: str) -> Tuple[int, int]:
    """
    Return (min_required_indicators, total_indicators_considered)
    A defensible default:
    - core (7): require >= 6
    - core+tertiary (8): require >= 7
    """
    total = len(indicators)
    if indicator_set == "core":
        return max(total - 1, 1), total
    if indicator_set == "core_plus_tertiary":
        return max(total - 1, 1), total
    return max(total - 1, 1), total


def compute_coverage_by_indicator(avg_df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    n = avg_df["country_iso3"].nunique()
    rows = []
    for ind in indicators:
        if ind not in avg_df.columns:
            rows.append({"indicator_code": ind, "n_countries_total": n, "n_non_missing": 0, "coverage_rate": 0.0})
            continue
        non_missing = avg_df[ind].notna().sum()
        rows.append({
            "indicator_code": ind,
            "n_countries_total": int(n),
            "n_non_missing": int(non_missing),
            "coverage_rate": float(non_missing / n) if n else None
        })
    return pd.DataFrame(rows).sort_values("coverage_rate", ascending=True)


def build_zscore(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    z = df.copy()
    for ind in indicators:
        if ind in z.columns:
            z[ind] = zscore_series(pd.to_numeric(z[ind], errors="coerce"))
    return z


# -----------------------------
# Main pipeline per spec
# -----------------------------

def run_spec(
    spec: Spec,
    panel: pd.DataFrame,
    cat: pd.DataFrame,
    outdir: Path,
    min_years_non_missing: int,
    winsor_p: float
) -> None:
    ensure_dir(outdir)

    indicators = select_indicators(cat, spec.indicator_set)
    year0, year1 = spec.time_window

    # 1) window avg + min-years rule per indicator
    avg_df, count_df = build_window_average(
        panel=panel,
        indicators=indicators,
        year0=year0,
        year1=year1,
        min_years_non_missing=min_years_non_missing
    )

    # 2) transforms (winsorize + log1p if configured)
    avg_tf = apply_transforms(avg_df, cat, indicators, winsor_p=winsor_p)

    # 3) inclusion rule: require at least N indicators present
    min_req, total = make_inclusion_rule(indicators, spec.indicator_set)
    present_count = avg_tf[indicators].notna().sum(axis=1)
    avg_tf["n_indicators_present"] = present_count
    avg_tf["is_included"] = present_count >= min_req
    avg_tf["inclusion_rule"] = f">= {min_req}/{total} indicators non-missing after min-year rule"

    included = avg_tf[avg_tf["is_included"]].copy()
    excluded = avg_tf[~avg_tf["is_included"]].copy()

    # 4) z-score dataset
    z_df = build_zscore(included, indicators)

    # 5) reports
    cov = compute_coverage_by_indicator(included, indicators)

    # output paths
    out_raw = outdir / "base_raw.csv"
    out_z = outdir / "base_zscore.csv"
    out_cov = outdir / "coverage_by_indicator.csv"
    out_inclusion = outdir / "inclusion_table.csv"
    out_report = outdir / "base_report.json"

    included.to_csv(out_raw, index=False)
    z_df.to_csv(out_z, index=False)
    cov.to_csv(out_cov, index=False)

    # Inclusion table includes both included/excluded with reasons
    inclusion_table = avg_tf[["country_iso3"] + [c for c in ["country_name", "region", "income_group", "lending_type"] if c in avg_tf.columns] +
                             ["n_indicators_present", "is_included", "inclusion_rule"]].copy()
    inclusion_table.to_csv(out_inclusion, index=False)

    report = {
        "spec": {
            "name": spec.name,
            "time_window": [year0, year1],
            "indicator_set": spec.indicator_set,
            "index_method": spec.index_method,
            "clustering_method": spec.clustering_method,
            "pca_level": spec.pca_level,
            "notes": spec.notes,
        },
        "params": {
            "min_years_non_missing": min_years_non_missing,
            "winsor_p": winsor_p,
            "inclusion_rule": f"keep countries with >= {min_req}/{total} indicators present after min-year rule",
        },
        "counts": {
            "countries_total_in_window": int(avg_df["country_iso3"].nunique()),
            "countries_included": int(included["country_iso3"].nunique()),
            "countries_excluded": int(excluded["country_iso3"].nunique()),
            "indicators_used": indicators,
        },
        "coverage": cov.to_dict(orient="records"),
    }

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{spec.name}] saved -> {outdir}")
    print(f"  Included countries: {report['counts']['countries_included']} / {report['counts']['countries_total_in_window']}")
    print(f"  Outputs: base_raw.csv, base_zscore.csv, coverage_by_indicator.csv, inclusion_table.csv, base_report.json")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build multi-spec base datasets for DII (window averaging, transforms, z-score).")
    ap.add_argument("--input_panel", type=str, required=True, help="Path to dii_panel_wide.csv")
    ap.add_argument("--catalog", type=str, required=True, help="Path to indicator_catalog.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory (spec folders will be created)")
    ap.add_argument("--spec", type=str, default="ALL", help="Spec to run: S1..S6 or ALL")
    ap.add_argument("--min_years_non_missing", type=int, default=DEFAULT_MIN_YEARS_NON_MISSING,
                    help="Minimum non-missing years per indicator within window (default: 3)")
    ap.add_argument("--winsor_p", type=float, default=DEFAULT_WINSOR_P,
                    help="Winsorize quantile p (default: 0.01)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    panel_path = Path(args.input_panel)
    catalog_path = Path(args.catalog)
    outroot = Path(args.outdir)

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)

    panel = pd.read_csv(panel_path)
    if "country_iso3" not in panel.columns or "year" not in panel.columns:
        raise ValueError("dii_panel_wide.csv phải có cột country_iso3 và year.")

    # enforce dtypes
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["country_iso3"] = panel["country_iso3"].astype(str)

    cat = load_catalog(catalog_path)

    # run selected specs
    spec_key = args.spec.upper().strip()
    if spec_key == "ALL":
        keys = ["S1", "S2", "S3", "S4", "S5", "S6"]
    else:
        if spec_key not in SPECS:
            raise ValueError(f"--spec phải là S1..S6 hoặc ALL. Bạn nhập: {args.spec}")
        keys = [spec_key]

    ensure_dir(outroot)

    for k in keys:
        spec = SPECS[k]
        spec_out = outroot / f"spec_{spec.name}"
        run_spec(
            spec=spec,
            panel=panel,
            cat=cat,
            outdir=spec_out,
            min_years_non_missing=args.min_years_non_missing,
            winsor_p=args.winsor_p
        )


if __name__ == "__main__":
    main()
