import yaml
import logging
from pathlib import Path
from typing import Dict, List


def resolve_out_dir(cfg: dict) -> Path:
    """Backward-compatible resolver for output directory key names."""
    paths = cfg.get("paths", {}) or {}
    for k in ["out_dir", "outdir", "out_dir_path", "outdir_path", "output_dir", "out"]:
        v = paths.get(k)
        if v:
            return Path(v)
    raise KeyError("Missing output dir in config. Expected one of: paths.out_dir/outdir/out_dir_path/output_dir")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s12_deep_build_lay")


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def cronbach_alpha(df: pd.DataFrame) -> float:
    x = df.dropna()
    if x.shape[1] < 2 or x.shape[0] < 5:
        return np.nan
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    k = x.shape[1]
    if total_var == 0 or np.isnan(total_var):
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def descriptive_table(s: pd.Series) -> dict:
    x = s.dropna().astype(float)
    if x.empty:
        return {k: np.nan for k in ["n","mean","sd","min","p5","median","p95","max","skew","kurtosis"]}
    return {
        "n": int(x.shape[0]),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=0)),
        "min": float(x.min()),
        "p5": float(np.percentile(x, 5)),
        "median": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
        "skew": float(stats.skew(x, bias=False)),
        "kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
    }


def main(config_path: str) -> None:
    cfg = read_yaml(Path(config_path))

    out_dir = resolve_out_dir(cfg)
    ensure_dir(out_dir / "panel")
    ensure_dir(out_dir / "audit_lay")

    build_cfg = cfg["build"]
    assert build_cfg["mode"] == "latest_available", "Script nay chi danh cho mode latest_available"

    win_start = int(build_cfg["window_start_year"])
    win_end = int(build_cfg["window_end_year"])
    edition_year = int(build_cfg["edition_year"])
    min_pillars_required = int(build_cfg["min_pillars_required"])

    # Load
    dii_panel = pd.read_csv(cfg["paths"]["dii_core_panel"])
    deep_long = pd.read_csv(out_dir / "raw_long" / "dii_deep_raw_long.csv")

    # Keep window
    deep_long["year"] = pd.to_numeric(deep_long["year"], errors="coerce").astype("Int64")
    deep_long["value"] = pd.to_numeric(deep_long["value"], errors="coerce")
    deep_long = deep_long.dropna(subset=["country_iso3", "year", "value", "indicator_code"])
    deep_long = deep_long[(deep_long["year"] >= win_start) & (deep_long["year"] <= win_end)].copy()

    if deep_long.empty:
        raise RuntimeError(f"Khong co du lieu deep trong cua so {win_start}-{win_end}. Hay doi window hoac them file ITU/OECD/Findex.")

    # Direction map and pillar map
    dir_map = (
        deep_long[["indicator_code", "direction"]]
        .drop_duplicates()
        .set_index("indicator_code")["direction"]
        .to_dict()
    )
    pillar_map = (
        deep_long[["indicator_code", "pillar"]]
        .drop_duplicates()
        .set_index("indicator_code")["pillar"]
        .to_dict()
    )

    # LAY: pick latest year per (country, indicator)
    deep_long = deep_long.sort_values(["country_iso3", "indicator_code", "year"])
    idx = deep_long.groupby(["country_iso3", "indicator_code"])["year"].idxmax()
    lay = deep_long.loc[idx, ["country_iso3", "indicator_code", "year", "value"]].copy()
    lay.rename(columns={"year": "ref_year", "value": "raw_value"}, inplace=True)

    # Add country_name/region/income_group anchor from DII panel (latest year within core range)
    # Use country metadata from any year in DII panel (stable)
    meta = (
        dii_panel[["country_iso3", "country_name", "region", "income_group", "lending_type"]]
        .drop_duplicates(subset=["country_iso3"])
        .copy()
    )
    lay = lay.merge(meta, on="country_iso3", how="left")

    # Pivot to wide: raw values + ref years
    val_wide = lay.pivot(index="country_iso3", columns="indicator_code", values="raw_value")
    yr_wide = lay.pivot(index="country_iso3", columns="indicator_code", values="ref_year")

    # Reverse if negative direction
    for code in val_wide.columns:
        if dir_map.get(code, "positive") == "negative":
            val_wide[code] = -val_wide[code]

    # Standardize to z across countries (cross-section)
    z_wide = val_wide.apply(zscore, axis=0)

    # Build pillars from indicator z
    all_codes = list(z_wide.columns)
    pillars = sorted(set(pillar_map.get(c, "unknown") for c in all_codes))
    pillar_scores = {}
    pillar_counts = {}
    for p in pillars:
        codes = [c for c in all_codes if pillar_map.get(c, "unknown") == p]
        if not codes:
            continue
        pillar_scores[f"deep_pillar_{p}"] = z_wide[codes].mean(axis=1, skipna=True)
        pillar_counts[f"deep_pillar_{p}_n"] = z_wide[codes].notna().sum(axis=1)

    pillar_df = pd.DataFrame(pillar_scores)
    pillar_n_df = pd.DataFrame(pillar_counts)

    # Index aggregation
    deep_n_pillars = pillar_df.notna().sum(axis=1)
    dii_deep_z = pillar_df.mean(axis=1, skipna=True)
    dii_deep_z[deep_n_pillars < min_pillars_required] = np.nan

    # Rescale to 0-100 on available sample
    def minmax_0_100(x: pd.Series) -> pd.Series:
        mn, mx = x.min(), x.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return x * np.nan
        return (x - mn) / (mx - mn) * 100.0

    dii_deep_0_100 = minmax_0_100(dii_deep_z)

    # Build final cross-section table
    out = pd.DataFrame({
        "country_iso3": val_wide.index,
    }).merge(meta, on="country_iso3", how="left")

    out = out.join(pillar_df, on="country_iso3")
    out = out.join(pillar_n_df, on="country_iso3")
    out["deep_n_pillars_available"] = deep_n_pillars.values
    out["dii_deep_z"] = dii_deep_z.values
    out["dii_deep_0_100"] = dii_deep_0_100.values

    # Add DII-Core snapshot for comparison (choose core year end=2022, country avg or 2022)
    core_2022 = dii_panel[dii_panel["year"] == 2022][["country_iso3", "dii_core_0_100"]].drop_duplicates()
    out = out.merge(core_2022, on="country_iso3", how="left")

    # Timeliness / staleness
    # country-level: median ref_year across indicators available
    yr_country = yr_wide.copy()
    out["ref_year_median"] = yr_country.median(axis=1, skipna=True).values
    out["ref_year_p10"] = yr_country.quantile(0.10, axis=1, interpolation="linear").values
    out["ref_year_p90"] = yr_country.quantile(0.90, axis=1, interpolation="linear").values
    out["staleness_median"] = (edition_year - out["ref_year_median"]).astype("Float64")

    # Save cross-section outputs
    cross_path = out_dir / "panel" / "dii_deep_lay_cross_section.csv"
    out.to_csv(cross_path, index=False)
    logger.info(f"[OK] Saved cross-section: {cross_path} | countries={out['country_iso3'].nunique()}")

    # Save indicator reference years matrix (useful for audit)
    yr_path = out_dir / "panel" / "dii_deep_lay_ref_year_matrix.csv"
    yr_wide.reset_index().to_csv(yr_path, index=False)

    # Save indicator values matrix (raw and z)
    val_path = out_dir / "panel" / "dii_deep_lay_raw_values_matrix.csv"
    val_wide.reset_index().to_csv(val_path, index=False)
    z_path = out_dir / "panel" / "dii_deep_lay_z_values_matrix.csv"
    z_wide.reset_index().to_csv(z_path, index=False)

    # ===== Audit LAY (quick but đủ để quyết định giữ/bỏ) =====
    audit_dir = out_dir / "audit_lay"

    # A) Timeliness by indicator
    tim_ind_rows = []
    for code in yr_wide.columns:
        ys = yr_wide[code].dropna().astype(float)
        if ys.empty:
            continue
        row = {
            "indicator_code": code,
            "pillar": pillar_map.get(code, "unknown"),
            "n": int(ys.shape[0]),
            "min_year": float(ys.min()),
            "p10": float(np.percentile(ys, 10)),
            "median": float(np.percentile(ys, 50)),
            "p90": float(np.percentile(ys, 90)),
            "max_year": float(ys.max()),
            "staleness_median": float(edition_year - np.percentile(ys, 50)),
        }
        # share by year
        for y in range(win_end, win_start - 1, -1):
            row[f"share_{y}"] = float((ys == y).mean())
        tim_ind_rows.append(row)

    tim_ind = pd.DataFrame(tim_ind_rows).sort_values(["pillar", "staleness_median"], ascending=[True, False])
    tim_ind.to_csv(audit_dir / "timeliness_by_indicator.csv", index=False)

    # B) Timeliness by country (staleness)
    tim_cty = out[["country_iso3", "country_name", "region", "income_group", "ref_year_median", "staleness_median"]].copy()
    tim_cty.to_csv(audit_dir / "timeliness_by_country.csv", index=False)

    # C) Missingness (indicator and pillar)
    miss_rows = []
    for c in z_wide.columns:
        miss_rows.append({
            "variable": c,
            "type": "indicator_z",
            "pillar": pillar_map.get(c, "unknown"),
            "n_non_missing": int(z_wide[c].notna().sum()),
            "missing_rate_pct": float(z_wide[c].isna().mean() * 100),
        })
    for c in pillar_df.columns:
        miss_rows.append({
            "variable": c,
            "type": "pillar",
            "pillar": c.replace("deep_pillar_", ""),
            "n_non_missing": int(pillar_df[c].notna().sum()),
            "missing_rate_pct": float(pillar_df[c].isna().mean() * 100),
        })
    miss = pd.DataFrame(miss_rows).sort_values(["type", "missing_rate_pct"], ascending=[True, False])
    miss.to_csv(audit_dir / "missingness.csv", index=False)

    # D) Descriptive stats
    desc_rows = []
    for c in list(pillar_df.columns) + ["dii_deep_0_100", "dii_core_0_100"]:
        s = out[c] if c in out.columns else pillar_df[c]
        row = {"variable": c}
        row.update(descriptive_table(s))
        desc_rows.append(row)
    pd.DataFrame(desc_rows).to_csv(audit_dir / "descriptives.csv", index=False)

    # E) Correlations (pillars + deep + core)
    corr_cols = list(pillar_df.columns) + ["dii_deep_0_100", "dii_core_0_100"]
    corr = out[corr_cols].corr(method="spearman")
    corr.to_csv(audit_dir / "spearman_corr.csv")

    # F) PCA on pillars (complete-case)
    pca_in = out[list(pillar_df.columns)].dropna()
    if pca_in.shape[0] >= 30 and pca_in.shape[1] >= 2:
        pca = PCA()
        X = pca_in.to_numpy(dtype=float)
        pca.fit(X)
        var_df = pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(pca_in.shape[1])],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        })
        var_df.to_csv(audit_dir / "pca_variance.csv", index=False)

        load_df = pd.DataFrame(
            pca.components_.T,
            index=list(pillar_df.columns),
            columns=[f"PC{i+1}" for i in range(pca_in.shape[1])]
        ).reset_index().rename(columns={"index": "variable"})
        load_df.to_csv(audit_dir / "pca_loadings.csv", index=False)

        # alpha (pillars)
        alpha = cronbach_alpha(out[list(pillar_df.columns)])
    else:
        alpha = np.nan

    pd.DataFrame([{
        "alpha_pillars": alpha,
        "n_complete_pillars": int(out[list(pillar_df.columns)].dropna().shape[0]),
    }]).to_csv(audit_dir / "reliability.csv", index=False)

    logger.info(f"[OK] LAY audit outputs saved at: {audit_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
