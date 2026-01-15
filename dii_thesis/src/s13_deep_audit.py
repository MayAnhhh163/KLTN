import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s13_deep_audit")


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_out_dir(cfg: dict) -> Path:
    """Backward-compatible resolver for output directory key names."""
    paths = cfg.get("paths", {}) or {}
    for k in ["out_dir", "outdir", "out_dir_path", "outdir_path", "output_dir", "out"]:
        v = paths.get(k)
        if v:
            return Path(v)
    raise KeyError("Missing output dir in config. Expected one of: paths.out_dir/outdir/out_dir_path/output_dir")


def zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def descriptive_table(s: pd.Series) -> dict:
    x = s.dropna().astype(float)
    if x.empty:
        return {k: np.nan for k in ["n", "mean", "sd", "min", "p5", "median", "p95", "max", "skew", "kurtosis"]}
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


def audit_lay(panel_path: Path, out_dir: Path, cfg: dict) -> None:
    """Audit for Latest-Available-Year (cross-section) build."""
    audit_dir = out_dir / "audit_lay"
    ensure_dir(audit_dir)

    df = pd.read_csv(panel_path)

    # Identify pillars
    pillar_cols = [c for c in df.columns if c.startswith("deep_pillar_") and not c.endswith("_n")]
    # Basic descriptives
    desc_rows = []
    for c in pillar_cols + [c for c in ["dii_deep_0_100", "dii_deep_z", "dii_core_0_100"] if c in df.columns]:
        desc_rows.append({"variable": c, **descriptive_table(df[c])})
    pd.DataFrame(desc_rows).to_csv(audit_dir / "descriptives.csv", index=False)

    # Missingness
    miss_rows = []
    for c in pillar_cols + ["dii_deep_0_100"]:
        if c not in df.columns:
            continue
        miss_rows.append({
            "variable": c,
            "type": "pillar" if c in pillar_cols else "index",
            "n_non_missing": int(df[c].notna().sum()),
            "missing_rate_pct": float(df[c].isna().mean() * 100),
        })
    pd.DataFrame(miss_rows).sort_values("missing_rate_pct", ascending=False).to_csv(audit_dir / "missingness.csv", index=False)

    # Correlations (Spearman + Pearson)
    corr_cols = [c for c in pillar_cols + ["dii_deep_0_100", "dii_core_0_100"] if c in df.columns]
    if len(corr_cols) >= 2:
        df[corr_cols].corr(method="spearman").to_csv(audit_dir / "spearman_corr.csv")
        df[corr_cols].corr(method="pearson").to_csv(audit_dir / "pearson_corr.csv")

    # Sanity: pillar vs core (Spearman)
    if "dii_core_0_100" in df.columns and pillar_cols:
        rows = []
        for p in pillar_cols:
            rho = df[[p, "dii_core_0_100"]].corr(method="spearman").iloc[0, 1]
            rows.append({"pillar": p, "spearman_with_core": rho})
        pd.DataFrame(rows).sort_values("spearman_with_core", ascending=False).to_csv(
            audit_dir / "sanity_pillar_vs_core.csv", index=False
        )

    # PCA on pillars (complete-case)
    pca_in = df[pillar_cols].dropna() if pillar_cols else pd.DataFrame()
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
            index=pillar_cols,
            columns=[f"PC{i+1}" for i in range(pca_in.shape[1])]
        ).reset_index().rename(columns={"index": "variable"})
        load_df.to_csv(audit_dir / "pca_loadings.csv", index=False)

        alpha = cronbach_alpha(df[pillar_cols])
    else:
        alpha = np.nan

    pd.DataFrame([{
        "alpha_pillars": alpha,
        "n_complete_pillars": int(df[pillar_cols].dropna().shape[0]) if pillar_cols else 0,
        "n_countries": int(df["country_iso3"].nunique()) if "country_iso3" in df.columns else int(df.shape[0]),
    }]).to_csv(audit_dir / "reliability.csv", index=False)

    # Sensitivity: leave-one-pillar-out (LOPO)
    if pillar_cols:
        base = df[pillar_cols].copy()
        base_score = base.mean(axis=1, skipna=True)
        base_rank = base_score.rank(ascending=False, method="average")
        rows = []
        for drop_p in pillar_cols:
            cols = [c for c in pillar_cols if c != drop_p]
            score = df[cols].mean(axis=1, skipna=True)
            r = score.rank(ascending=False, method="average")
            rho = pd.concat([base_rank, r], axis=1).corr(method="spearman").iloc[0, 1]
            rows.append({"dropped_pillar": drop_p, "spearman_rank_vs_full": rho})
        pd.DataFrame(rows).sort_values("spearman_rank_vs_full").to_csv(
            audit_dir / "sensitivity_leave_one_pillar_out.csv", index=False
        )

    # Outliers: rank gap deep vs core
    if "dii_core_0_100" in df.columns and "dii_deep_0_100" in df.columns:
        tmp = df[["country_iso3", "country_name", "region", "income_group", "dii_deep_0_100", "dii_core_0_100"]].copy()
        tmp["rank_deep"] = tmp["dii_deep_0_100"].rank(ascending=False, method="min")
        tmp["rank_core"] = tmp["dii_core_0_100"].rank(ascending=False, method="min")
        tmp["rank_gap_deep_minus_core"] = tmp["rank_deep"] - tmp["rank_core"]
        tmp["abs_rank_gap"] = tmp["rank_gap_deep_minus_core"].abs()
        tmp = tmp.sort_values("abs_rank_gap", ascending=False)
        tmp.head(50).to_csv(audit_dir / "outliers_rank_gap_deep_vs_core.csv", index=False)

    # Timeliness: only if ref-year matrix exists
    ref_year_matrix = out_dir / "panel" / "dii_deep_lay_ref_year_matrix.csv"
    if ref_year_matrix.exists():
        yrs = pd.read_csv(ref_year_matrix)
        if "country_iso3" in yrs.columns:
            yr_cols = [c for c in yrs.columns if c != "country_iso3"]
            tim_rows = []
            edition_year = int((cfg.get("build", {}) or {}).get("edition_year", (cfg.get("build", {}) or {}).get("window_end_year", 2025)))
            for c in yr_cols:
                ys = pd.to_numeric(yrs[c], errors="coerce").dropna()
                if ys.empty:
                    continue
                row = {
                    "indicator_code": c,
                    "n": int(ys.shape[0]),
                    "min_year": float(ys.min()),
                    "p10": float(np.percentile(ys, 10)),
                    "median": float(np.percentile(ys, 50)),
                    "p90": float(np.percentile(ys, 90)),
                    "max_year": float(ys.max()),
                    "staleness_median": float(edition_year - np.percentile(ys, 50)),
                }
                tim_rows.append(row)
            pd.DataFrame(tim_rows).sort_values("staleness_median", ascending=False).to_csv(
                audit_dir / "timeliness_by_indicator.csv", index=False
            )

    logger.info(f"[OK] LAY audit saved at: {audit_dir}")


def audit_panel(panel_path: Path, out_dir: Path) -> None:
    """Audit for strict country-year panel build (legacy)."""
    audit_dir = out_dir / "audit"
    ensure_dir(audit_dir)

    panel = pd.read_csv(panel_path)
    pillar_cols = [c for c in panel.columns if c.startswith("deep_pillar_") and not c.endswith("_n")]
    ind_z_cols = [c for c in panel.columns if c.endswith("_z") and not c.startswith("dii_")]

    cov = panel.groupby("year").agg(
        n_countries=("country_iso3", "nunique"),
        n_with_deep=("dii_deep_0_100", lambda x: x.notna().sum()),
        n_missing_deep=("dii_deep_0_100", lambda x: x.isna().sum()),
    ).reset_index()
    cov.to_csv(audit_dir / "audit_coverage_by_year.csv", index=False)

    year_latest = int(panel["year"].max())
    cross = panel[panel["year"] == year_latest].copy()

    rows = []
    for c in pillar_cols + ind_z_cols + ["dii_deep_0_100"]:
        if c not in cross.columns:
            continue
        rows.append({
            "variable": c,
            "n_non_missing": int(cross[c].notna().sum()),
            "missing_rate_pct": float(cross[c].isna().mean() * 100),
        })
    pd.DataFrame(rows).sort_values("missing_rate_pct", ascending=False).to_csv(
        audit_dir / "missingness_latest_year.csv", index=False
    )

    corr_cols = [c for c in pillar_cols + ["dii_deep_0_100"] if c in cross.columns]
    if len(corr_cols) >= 2:
        cross[corr_cols].corr(method="spearman").to_csv(audit_dir / "spearman_corr_latest_year.csv")

    pca_in = cross[pillar_cols].dropna()
    if pca_in.shape[0] >= 30 and pca_in.shape[1] >= 2:
        pca = PCA()
        X = pca_in.to_numpy(dtype=float)
        pca.fit(X)
        pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(pca_in.shape[1])],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }).to_csv(audit_dir / "pca_variance_latest_year.csv", index=False)

        pd.DataFrame(
            pca.components_.T,
            index=pillar_cols,
            columns=[f"PC{i+1}" for i in range(pca_in.shape[1])]
        ).reset_index().rename(columns={"index": "variable"}).to_csv(
            audit_dir / "pca_loadings_latest_year.csv", index=False
        )

    logger.info(f"[OK] Panel audit saved at: {audit_dir}")


def main(config_path: str) -> None:
    cfg = read_yaml(Path(config_path))
    out_dir = resolve_out_dir(cfg)

    panel_panel = out_dir / "panel" / "dii_deep_panel.csv"
    panel_lay = out_dir / "panel" / "dii_deep_lay_cross_section.csv"

    if panel_panel.exists():
        audit_panel(panel_panel, out_dir)
    elif panel_lay.exists():
        audit_lay(panel_lay, out_dir, cfg)
    else:
        raise FileNotFoundError(
            f"Cannot find deep outputs. Expected either:\n- {panel_panel}\n- {panel_lay}"
        )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
