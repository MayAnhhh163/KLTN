# s6_build_index.py
# Step 6: Build Digital Inclusion Index (DII) for multi-spec framework
#
# Inputs (from Step 5):
#   <spec_dir>/base_zscore.csv
#   <spec_dir>/base_raw.csv
#   <spec_dir>/base_report.json
#
# Outputs (per spec):
#   index_scores.csv          : DII scores + optional pillar scores
#   pca_loadings.csv          : PCA loadings (global or per pillar)
#   pca_summary.json          : explained variance + sign flip info
#   index_diagnostics.csv     : simple validity checks (correlations)
#
# Run:
#    python dii_thesis/src/s6_build_index.py \
#   --spec_root dii_thesis/data/processed/spec_outputs \
#   --catalog dii_thesis/config/indicator_catalog.csv \
#   --spec ALL
#
# Notes:
# - Deterministic.
# - Uses sklearn PCA.
# - Designed for research defensibility: sign alignment, scaling, and diagnostics.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# -----------------------------
# Specs (must match Step 5)
# -----------------------------
@dataclass(frozen=True)
class Spec:
    name: str
    indicator_set: str
    index_method: str
    pca_level: str  # "global" or "pillar"


SPECS: Dict[str, Spec] = {
    "S1": Spec(name="S1", indicator_set="core", index_method="pca", pca_level="global"),
    "S2": Spec(name="S2", indicator_set="core", index_method="equal_weight", pca_level="global"),
    "S3": Spec(name="S3", indicator_set="core", index_method="pca", pca_level="global"),
    "S4": Spec(name="S4", indicator_set="core", index_method="pca", pca_level="global"),
    "S5": Spec(name="S5", indicator_set="core", index_method="pca", pca_level="pillar"),
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    cat = pd.read_csv(catalog_path)
    needed = {"indicator_code", "pillar", "priority"}
    missing = needed - set(cat.columns)
    if missing:
        raise ValueError(f"indicator_catalog.csv thiếu cột: {sorted(missing)}")
    cat["indicator_code"] = cat["indicator_code"].astype(str)
    cat["pillar"] = cat["pillar"].astype(str)
    cat["priority"] = pd.to_numeric(cat["priority"], errors="coerce").fillna(999).astype(int)
    return cat


def select_indicators(cat: pd.DataFrame, indicator_set: str) -> List[str]:
    core = cat.loc[cat["priority"] == 1, "indicator_code"].tolist()
    if indicator_set == "core":
        return core
    if indicator_set == "core_plus_tertiary":
        optional = cat.loc[cat["priority"] >= 2, "indicator_code"].tolist()
        return core + [x for x in optional if x not in core]
    raise ValueError(f"indicator_set không hợp lệ: {indicator_set}")


def minmax_0_100(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    lo = x.min(skipna=True)
    hi = x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return x.apply(lambda v: 50.0 if pd.notna(v) else np.nan)
    return (x - lo) / (hi - lo) * 100.0


def align_sign(score: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """
    Align sign of a 1D score so it correlates positively with reference.
    Returns (score_aligned, sign_flip, corr).
    sign_flip = 1 if flipped, else 0.
    """
    s = pd.Series(score)
    r = pd.Series(reference)
    corr = s.corr(r)
    if pd.isna(corr):
        return score, 0, float("nan")
    if corr < 0:
        return -score, 1, float(-corr)
    return score, 0, float(corr)


def pca_one_component(X: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=1, random_state=0)
    pc1 = pca.fit_transform(X).reshape(-1)
    return pc1, pca


def safe_numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """
    Return numeric matrix for sklearn.
    Research-defensible handling:
    - If NaN exists in z-scored indicators, impute with 0.0 (the mean in z-score space),
      which preserves sample size and avoids dropping countries.
    - This is appropriate because Step 5 already enforces a minimum number of present indicators.
    """
    Xdf = df[cols].apply(pd.to_numeric, errors="coerce")

    n_missing = int(Xdf.isna().sum().sum())
    if n_missing > 0:
        # Impute missing z-scores with 0 (mean) to keep countries in global sample
        Xdf = Xdf.fillna(0.0)

    return Xdf.to_numpy(dtype=float)

# -----------------------------
# Index builders
# -----------------------------
def build_equal_weight(df_z: pd.DataFrame, indicators: List[str]) -> Dict[str, pd.Series]:
    # Equal-weight on z-scores: mean across indicators
    ew = df_z[indicators].mean(axis=1)
    ew_0100 = minmax_0_100(ew)
    return {
        "DII_equal_zmean": ew,
        "DII_equal_0100": ew_0100
    }


def build_global_pca(df_z: pd.DataFrame, indicators: List[str]) -> Tuple[Dict[str, pd.Series], pd.DataFrame, Dict]:
    # reference: mean of z-scores (monotonic with "more inclusion")
    ref = df_z[indicators].mean(axis=1).to_numpy()

    X = safe_numeric_matrix(df_z, indicators)
    pc1, pca = pca_one_component(X)

    pc1_aligned, flipped, corr = align_sign(pc1, ref)
    dii_0100 = minmax_0_100(pd.Series(pc1_aligned))

    # loadings: sklearn components_ are eigenvectors; for standardized inputs, loadings ~ components_
    loadings = pd.DataFrame({
        "indicator_code": indicators,
        "pc1_loading": pca.components_[0]
    })

    summary = {
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "sign_flipped": int(flipped),
        "corr_with_reference_mean_z": float(corr),
    }

    out = {
        "DII_pca_pc1": pd.Series(pc1_aligned),
        "DII_pca_0100": dii_0100
    }

    return out, loadings, summary


def build_pillar_pca(df_z: pd.DataFrame, cat: pd.DataFrame, indicators: List[str]) -> Tuple[Dict[str, pd.Series], pd.DataFrame, Dict]:
    """
    Pillar-level PCA:
    - For each pillar: PCA(1) on indicators in that pillar -> pillar_score (aligned)
    - DII_pillar = mean(pillar_scores) -> scaled 0-100
    Outputs:
    - scores dict: pillar scores + DII
    - loadings df: per pillar loadings
    - summary dict
    """
    # Map indicator -> pillar
    pillar_map = cat.set_index("indicator_code")["pillar"].to_dict()

    # Group indicators by pillar (only those in current spec)
    pillars: Dict[str, List[str]] = {}
    for ind in indicators:
        p = pillar_map.get(ind, "unknown")
        pillars.setdefault(p, []).append(ind)

    # Use only expected pillars (if unknown present, keep but note)
    load_rows = []
    pillar_scores = {}
    pillar_summaries = {}

    for pillar, inds in pillars.items():
        if len(inds) == 1:
            # If only one indicator in a pillar, use it directly (already z-scored)
            raw_score = df_z[inds[0]].to_numpy()
            ref = df_z[inds].mean(axis=1).to_numpy()
            aligned, flipped, corr = align_sign(raw_score, ref)
            pillar_scores[pillar] = pd.Series(aligned)
            pillar_summaries[pillar] = {
                "method": "single_indicator",
                "indicator": inds[0],
                "sign_flipped": int(flipped),
                "corr_with_reference": float(corr)
            }
            load_rows.append({"pillar": pillar, "indicator_code": inds[0], "pc1_loading": 1.0})
            continue

        Xp = safe_numeric_matrix(df_z, inds)
        pc1, pca = pca_one_component(Xp)
        ref = df_z[inds].mean(axis=1).to_numpy()
        pc1_aligned, flipped, corr = align_sign(pc1, ref)

        pillar_scores[pillar] = pd.Series(pc1_aligned)
        pillar_summaries[pillar] = {
            "method": "pca_pc1",
            "n_indicators": int(len(inds)),
            "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
            "sign_flipped": int(flipped),
            "corr_with_reference": float(corr)
        }

        for ind, loading in zip(inds, pca.components_[0]):
            load_rows.append({"pillar": pillar, "indicator_code": ind, "pc1_loading": float(loading)})

    # Build DII from pillar scores: equal-weight mean across main pillars
    # Expect pillars names in your catalog: access_use, quality_infra, skills (+ maybe affordability_proxy)
    pillar_cols = []
    for p in ["access_use", "quality_infra", "skills"]:
        if p in pillar_scores:
            pillar_cols.append(p)

    if not pillar_cols:
        # fallback: use all pillars present
        pillar_cols = list(pillar_scores.keys())

    pillar_matrix = np.column_stack([pillar_scores[p].to_numpy() for p in pillar_cols])
    dii_pillar = pd.Series(pillar_matrix.mean(axis=1))
    dii_pillar_0100 = minmax_0_100(dii_pillar)

    scores_out = {
        "DII_pillar_mean": dii_pillar,
        "DII_pillar_0100": dii_pillar_0100
    }
    # also export pillar scores individually
    for p, s in pillar_scores.items():
        scores_out[f"pillar_{p}_pc1"] = s

    loadings_df = pd.DataFrame(load_rows)

    summary = {
        "pillars_used_in_dii": pillar_cols,
        "pillar_summaries": pillar_summaries
    }

    return scores_out, loadings_df, summary


# -----------------------------
# Diagnostics
# -----------------------------
def build_diagnostics(df: pd.DataFrame, indicators: List[str], score_cols: List[str]) -> pd.DataFrame:
    """
    Simple external validity checks:
    - Correlation between each DII score and each indicator
    - Also correlation among DII scores
    """
    rows = []

    # DII vs indicators
    for sc in score_cols:
        for ind in indicators:
            if sc in df.columns and ind in df.columns:
                rows.append({
                    "type": "score_vs_indicator",
                    "score": sc,
                    "indicator": ind,
                    "corr": float(pd.Series(df[sc]).corr(pd.Series(df[ind])))
                })

    # DII vs DII
    for i in range(len(score_cols)):
        for j in range(i + 1, len(score_cols)):
            a, b = score_cols[i], score_cols[j]
            if a in df.columns and b in df.columns:
                rows.append({
                    "type": "score_vs_score",
                    "score": a,
                    "indicator": b,
                    "corr": float(pd.Series(df[a]).corr(pd.Series(df[b])))
                })

    return pd.DataFrame(rows)


# -----------------------------
# Runner per spec
# -----------------------------
def run_spec(spec_root: Path, spec: Spec, cat: pd.DataFrame) -> None:
    spec_dir = spec_root / f"spec_{spec.name}"
    if not spec_dir.exists():
        raise FileNotFoundError(f"Không thấy thư mục: {spec_dir}. Bạn đã chạy Step 5 chưa?")

    z_path = spec_dir / "base_zscore.csv"
    raw_path = spec_dir / "base_raw.csv"

    if not z_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Thiếu base_zscore.csv hoặc base_raw.csv trong {spec_dir}")

    df_z = pd.read_csv(z_path)
    df_raw = pd.read_csv(raw_path)

    indicators = select_indicators(cat, spec.indicator_set)

    # ensure all indicators exist
    missing_cols = [c for c in indicators if c not in df_z.columns]
    if missing_cols:
        raise ValueError(f"{spec.name}: base_zscore.csv thiếu các cột indicators: {missing_cols}")

    # Build indices
    outputs_scores: Dict[str, pd.Series] = {}
    loadings_all = []
    summary = {"spec": spec.name, "indicator_set": spec.indicator_set, "index_method": spec.index_method, "pca_level": spec.pca_level}

    # Always compute equal-weight (for diagnostics), but mark as primary only for S2
    ew = build_equal_weight(df_z, indicators)
    outputs_scores.update(ew)

    if spec.pca_level == "global":
        pca_scores, loadings, pca_sum = build_global_pca(df_z, indicators)
        outputs_scores.update(pca_scores)
        loadings["scope"] = "global"
        loadings_all.append(loadings)
        summary["global_pca"] = pca_sum
    else:
        # pillar-level PCA
        pillar_scores, pillar_loadings, pillar_sum = build_pillar_pca(df_z, cat, indicators)
        outputs_scores.update(pillar_scores)
        pillar_loadings["scope"] = "pillar"
        loadings_all.append(pillar_loadings)
        summary["pillar_pca"] = pillar_sum

    # Merge scores back to raw for interpretability (raw already transformed)
    # Keep key metadata columns if present
    meta_cols = [c for c in ["country_iso3", "country_name", "region", "income_group", "lending_type"] if c in df_raw.columns]
    out_df = df_raw[meta_cols + indicators].copy()

    for k, s in outputs_scores.items():
        out_df[k] = s.values

    # Determine "primary" DII for the spec (used later in clustering)
    if spec.name == "S2":
        primary = "DII_equal_0100"
    elif spec.name == "S5":
        primary = "DII_pillar_0100"
    else:
        primary = "DII_pca_0100"
    out_df["DII_primary"] = out_df[primary]

    # Diagnostics
    score_cols = [c for c in ["DII_pca_0100", "DII_equal_0100", "DII_pillar_0100"] if c in out_df.columns]
    diag = build_diagnostics(out_df, indicators, score_cols)

    # Save outputs
    out_scores = spec_dir / "index_scores.csv"
    out_loadings = spec_dir / "pca_loadings.csv"
    out_summary = spec_dir / "pca_summary.json"
    out_diag = spec_dir / "index_diagnostics.csv"

    out_df.to_csv(out_scores, index=False)

    if loadings_all:
        pd.concat(loadings_all, ignore_index=True).to_csv(out_loadings, index=False)
    else:
        pd.DataFrame().to_csv(out_loadings, index=False)

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    diag.to_csv(out_diag, index=False)

    print(f"[{spec.name}] saved: {out_scores.name}, {out_loadings.name}, {out_summary.name}, {out_diag.name}")
    print(f"  Primary DII: {primary}  | rows={len(out_df)}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 6: Build DII indices (PCA / equal-weight / pillar PCA) for multi-spec.")
    ap.add_argument("--spec_root", type=str, required=True, help="Root folder containing spec_S1..spec_S5 from Step 5")
    ap.add_argument("--catalog", type=str, required=True, help="Path to indicator_catalog.csv")
    ap.add_argument("--spec", type=str, default="ALL", help="S1..S5 or ALL")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    spec_root = Path(args.spec_root)
    catalog_path = Path(args.catalog)

    if not spec_root.exists():
        raise FileNotFoundError(spec_root)
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)

    cat = load_catalog(catalog_path)

    key = args.spec.upper().strip()
    if key == "ALL":
        keys = ["S1", "S2", "S3", "S4", "S5"]
    else:
        if key not in SPECS:
            raise ValueError("--spec phải là S1..S5 hoặc ALL")
        keys = [key]

    for k in keys:
        run_spec(spec_root, SPECS[k], cat)


if __name__ == "__main__":
    main()
