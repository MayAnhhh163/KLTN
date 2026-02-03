# s6_build_index.py
# Step 6 (UPDATED): Tạo các bảng tổng hợp phục vụ viết luận văn từ DII-Core panel
#
# Input:
#   - data/processed/dii_core/dii_core_panel.csv
#
# Output (data/processed/dii_core/):
#   - dii_core_year_summary.csv        : thống kê theo năm (mean/median/coverage)
#   - dii_core_income_summary.csv      : thống kê theo năm & nhóm thu nhập
#   - dii_core_region_summary.csv      : thống kê theo năm & vùng
#   - dii_core_year_ranking_top50.csv  : top 50 theo từng năm (0–100 scale)
#   - dii_core_year_ranking_bottom50.csv: bottom 50 theo từng năm
#
# Ghi chú:
# - Script này KHÔNG thay đổi logic tính chỉ số. Nó chỉ tạo bảng để bạn đưa vào chương Kết quả.
# - Nếu thiếu region/income_group trong panel, các bảng tương ứng sẽ tự bỏ qua.

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from dii_thesis.src.s0_settings import PROCESSED_DIR, STUDY_START_YEAR, STUDY_END_YEAR


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["has_dii"] = d["dii_core_0_100"].notna()
    out = (
        d.groupby("year", as_index=False)
        .agg(
            n_rows=("country_iso3", "size"),
            n_countries=("country_iso3", "nunique"),
            share_has_dii=("has_dii", "mean"),
            mean_dii=("dii_core_0_100", "mean"),
            median_dii=("dii_core_0_100", "median"),
            p25=("dii_core_0_100", lambda s: s.quantile(0.25)),
            p75=("dii_core_0_100", lambda s: s.quantile(0.75)),
        )
        .sort_values("year")
    )
    return out


def build_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    d = df.copy()
    d["has_dii"] = d["dii_core_0_100"].notna()
    out = (
        d.dropna(subset=[group_col])
        .groupby(["year", group_col], as_index=False)
        .agg(
            n_countries=("country_iso3", "nunique"),
            share_has_dii=("has_dii", "mean"),
            mean_dii=("dii_core_0_100", "mean"),
            median_dii=("dii_core_0_100", "median"),
        )
        .sort_values(["year", group_col])
    )
    return out


def build_year_rankings(df: pd.DataFrame, top_n: int, ascending: bool) -> pd.DataFrame:
    rows = []
    for y, dy in df.groupby("year"):
        dy = dy.dropna(subset=["dii_core_0_100"]).copy()
        dy = dy.sort_values("dii_core_0_100", ascending=ascending).head(top_n)

        # rank theo thứ tự sau khi sort
        dy["rank"] = range(1, len(dy) + 1)

        # dy đã có cột year rồi, nên chỉ cần chọn cột xuất ra
        rows.append(dy[["year", "rank", "country_iso3", "country_name", "dii_core_0_100"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["year", "rank", "country_iso3", "country_name", "dii_core_0_100"]
    )


def main(panel_path: Path, outdir: Path, top_n: int = 50) -> None:
    _ensure_dir(outdir)

    df = pd.read_csv(panel_path)

    # Defensive: lọc giai đoạn
    df = df[(df["year"] >= STUDY_START_YEAR) & (df["year"] <= STUDY_END_YEAR)].copy()

    # Year summary
    build_year_summary(df).to_csv(outdir / "dii_core_year_summary.csv", index=False)

    # Income/Region summaries (nếu có)
    if "income_group" in df.columns:
        build_group_summary(df, "income_group").to_csv(outdir / "dii_core_income_summary.csv", index=False)

    if "region" in df.columns:
        build_group_summary(df, "region").to_csv(outdir / "dii_core_region_summary.csv", index=False)

    # Rankings
    cols_needed = ["country_iso3", "country_name", "year", "dii_core_0_100"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột {c} trong dii_core_panel.csv")

    build_year_rankings(df, top_n=top_n, ascending=False).to_csv(outdir / "dii_core_year_ranking_top50.csv", index=False)
    build_year_rankings(df, top_n=top_n, ascending=True).to_csv(outdir / "dii_core_year_ranking_bottom50.csv", index=False)

    print(f"Saved thesis tables to: {outdir}")


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
        default=str(PROCESSED_DIR / "dii_core"),
        help="Output directory.",
    )
    parser.add_argument("--top_n", type=int, default=50)
    args = parser.parse_args()
    main(Path(args.panel), Path(args.outdir), top_n=args.top_n)
