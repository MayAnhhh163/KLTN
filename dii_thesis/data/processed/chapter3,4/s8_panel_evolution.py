# s8_panel_evolution.py
# Step 8 (UPDATED): Tổng hợp diễn biến DII-Core theo thời gian (2015–2022)
#
# Input:
#   - data/processed/dii_core/dii_core_panel.csv
#
# Output (data/processed/dii_core/evolution/):
#   - evolution_global.csv               : thống kê global theo năm
#   - evolution_by_income_group.csv      : theo năm & nhóm thu nhập (nếu có)
#   - evolution_by_region.csv            : theo năm & vùng (nếu có)
#   - plot_global_mean.png               : biểu đồ mean DII theo năm (nếu matplotlib chạy được)
#
# Ghi chú:
# - Script này phục vụ chương "Kết quả" và "Thảo luận", giúp bạn kể câu chuyện động lực 2015–2022.
# - Không thay đổi logic tính chỉ số.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from dii_thesis.src.s0_settings import PROCESSED_DIR, STUDY_START_YEAR, STUDY_END_YEAR


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def evolution_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["has_dii"] = d["dii_core_0_100"].notna()

    agg = {
        "n_countries": ("country_iso3", "nunique"),
        "share_has_dii": ("has_dii", "mean"),
        "mean_dii": ("dii_core_0_100", "mean"),
        "median_dii": ("dii_core_0_100", "median"),
        "p25": ("dii_core_0_100", lambda s: s.quantile(0.25)),
        "p75": ("dii_core_0_100", lambda s: s.quantile(0.75)),
    }
    out = (
        d.groupby(group_cols, as_index=False)
        .agg(**{k: v for k, v in agg.items()})
        .sort_values(group_cols)
    )
    return out


def plot_global_mean(global_tbl: pd.DataFrame, out_png: Path) -> None:
    # Không set màu theo yêu cầu; matplotlib dùng mặc định
    plt.figure(figsize=(8, 4.5))
    plt.plot(global_tbl["year"], global_tbl["mean_dii"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean DII-Core (0-100)")
    plt.title("Global Mean DII-Core (2015–2022)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main(panel_path: Path, outdir: Path) -> None:
    _ensure_dir(outdir)

    df = pd.read_csv(panel_path)

    # Defensive: lọc giai đoạn
    df = df[(df["year"] >= STUDY_START_YEAR) & (df["year"] <= STUDY_END_YEAR)].copy()

    # Global
    global_tbl = evolution_table(df, ["year"])
    global_tbl.to_csv(outdir / "evolution_global.csv", index=False)

    # Income group (nếu có)
    if "income_group" in df.columns:
        inc_tbl = df.dropna(subset=["income_group"])
        evolution_table(inc_tbl, ["year", "income_group"]).to_csv(outdir / "evolution_by_income_group.csv", index=False)

    # Region (nếu có)
    if "region" in df.columns:
        reg_tbl = df.dropna(subset=["region"])
        evolution_table(reg_tbl, ["year", "region"]).to_csv(outdir / "evolution_by_region.csv", index=False)

    # Plot
    plot_global_mean(global_tbl, outdir / "plot_global_mean.png")

    print(f"Saved evolution outputs to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=str,
        default=str(PROCESSED_DIR / "dii_core" / "dii_core_panel.csv"),
        help="Path tới dii_core_panel.csv (Step 5 output).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROCESSED_DIR / "dii_core" / "evolution"),
        help="Output directory.",
    )
    args = parser.parse_args()
    main(Path(args.panel), Path(args.outdir))
