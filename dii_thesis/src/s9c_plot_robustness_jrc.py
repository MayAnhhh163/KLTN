from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


def _savefig(fig: plt.Figure, outpath: Path, dpi: int = 220) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_equal_vs_pca(csv_path: Path, out_png: Path) -> None:
    df = pd.read_csv(csv_path)
    _require_cols(df, ["rank_equal", "rank_pca"], "weights_pca_vs_equal")

    x = pd.to_numeric(df["rank_equal"], errors="coerce")
    y = pd.to_numeric(df["rank_pca"], errors="coerce")
    m = x.notna() & y.notna()

    x = x[m].to_numpy()
    y = y[m].to_numpy()

    # 45-degree line bounds
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=12, alpha=0.65)
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_title("Rank comparison: Equal weights vs PCA-based weights")
    ax.set_xlabel("Rank (Equal weights)")
    ax.set_ylabel("Rank (PCA-based weights)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    _savefig(fig, out_png)


def plot_scatter_arith_vs_geom(csv_path: Path, out_png: Path) -> None:
    df = pd.read_csv(csv_path)
    need = ["rank_arithmetic", "rank_geometric", "delta_rank"]
    _require_cols(df, need, "aggregation_geom_vs_arith")

    x = pd.to_numeric(df["rank_arithmetic"], errors="coerce")
    y = pd.to_numeric(df["rank_geometric"], errors="coerce")
    d = pd.to_numeric(df["delta_rank"], errors="coerce")
    m = x.notna() & y.notna() & d.notna()

    dfp = df.loc[m].copy()
    dfp["rank_arithmetic"] = x[m].values
    dfp["rank_geometric"] = y[m].values
    dfp["delta_rank"] = d[m].values
    dfp["abs_delta"] = dfp["delta_rank"].abs()

    # highlight the single largest |delta_rank| point (optional annotation)
    idx = dfp["abs_delta"].idxmax()
    row = dfp.loc[idx]

    lo = float(min(dfp["rank_arithmetic"].min(), dfp["rank_geometric"].min()))
    hi = float(max(dfp["rank_arithmetic"].max(), dfp["rank_geometric"].max()))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(dfp["rank_arithmetic"], dfp["rank_geometric"], s=12, alpha=0.65)
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_title("Rank comparison: Arithmetic vs Geometric-like aggregation")
    ax.set_xlabel("Rank (Arithmetic)")
    ax.set_ylabel("Rank (Geometric-like)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # annotate the max-change country if name/iso exists
    label_parts = []
    if "country_name" in dfp.columns and pd.notna(row.get("country_name")):
        label_parts.append(str(row["country_name"]))
    if "country_iso3" in dfp.columns and pd.notna(row.get("country_iso3")):
        label_parts.append(str(row["country_iso3"]))
    label = " | ".join(label_parts) if label_parts else "max |Δrank|"

    ax.annotate(
        f"{label}\nΔrank={int(row['delta_rank'])}",
        xy=(row["rank_arithmetic"], row["rank_geometric"]),
        xytext=(row["rank_arithmetic"] + 5, row["rank_geometric"] + 5),
        arrowprops=dict(arrowstyle="->", linewidth=1),
        fontsize=9,
    )

    _savefig(fig, out_png)


def plot_boxplot_leave_one_out(csv_path: Path, out_png: Path) -> None:
    df = pd.read_csv(csv_path)
    _require_cols(df, ["indicator_removed", "delta_rank"], "leave_one_out_country")

    df["abs_delta_rank"] = pd.to_numeric(df["delta_rank"], errors="coerce").abs()
    df = df.dropna(subset=["indicator_removed", "abs_delta_rank"]).copy()

    # stable ordering: by median |Δrank| descending (more informative)
    order = (
        df.groupby("indicator_removed")["abs_delta_rank"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    data = [df.loc[df["indicator_removed"] == k, "abs_delta_rank"].to_numpy() for k in order]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.boxplot(data, vert=True, showfliers=True)
    ax.set_title("Leave-one-out sensitivity: |Δrank| by removed indicator")
    ax.set_ylabel("|Δrank|")
    ax.set_xlabel("Removed indicator")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right")

    _savefig(fig, out_png)


def main(in_dir: Path, out_dir: Path) -> None:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f1 = in_dir / "robustness_weights_pca_vs_equal_country.csv"
    f2 = in_dir / "robustness_aggregation_geom_vs_arith_country.csv"
    f3 = in_dir / "robustness_leave_one_out_country.csv"

    if not f1.exists():
        raise FileNotFoundError(f"Missing: {f1}")
    if not f2.exists():
        raise FileNotFoundError(f"Missing: {f2}")
    if not f3.exists():
        raise FileNotFoundError(f"Missing: {f3}")

    plot_scatter_equal_vs_pca(f1, out_dir / "fig_5x_scatter_rank_equal_vs_pca.png")
    plot_scatter_arith_vs_geom(f2, out_dir / "fig_5z_scatter_rank_arith_vs_geom.png")
    plot_boxplot_leave_one_out(f3, out_dir / "fig_5y_boxplot_leave_one_out_abs_delta_rank.png")

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_dir",
        type=str,
        default=str(Path("dii_thesis") / "data" / "processed" / "dii_core" / "robustness_jrc"),
        help="Folder containing robustness_jrc CSV outputs",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("dii_thesis") / "data" / "processed" / "dii_core" / "robustness_jrc"),
        help="Folder to save PNG figures (default: same as in_dir)",
    )
    args = p.parse_args()
    main(Path(args.in_dir), Path(args.out_dir))
