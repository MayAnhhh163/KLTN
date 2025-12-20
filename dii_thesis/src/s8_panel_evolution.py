import pandas as pd
from pathlib import Path

from dii_thesis.src.s0_settings import PROCESSED_DIR, CONFIG_DIR


def main():
    # Load panel wide (2010–2023)
    panel = pd.read_csv(PROCESSED_DIR / "dii_panel_wide.csv")

    # Load catalog để xác định core indicators
    catalog = pd.read_csv(CONFIG_DIR / "indicator_catalog.csv")
    core_inds = (
        catalog[catalog["priority"] == 1]["indicator_code"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Giữ các cột cần thiết
    keep_cols = ["country_iso3", "year"] + core_inds
    panel = panel[keep_cols].copy()

    # Chỉ giữ các năm có ít nhất 1 indicator non-missing
    panel = panel.dropna(subset=["year"])

    # Chuẩn hóa theo năm (z-score cross-section mỗi năm)
    z_panels = []
    for y, df_y in panel.groupby("year"):
        df_y = df_y.copy()
        for ind in core_inds:
            if ind in df_y.columns:
                mu = df_y[ind].mean(skipna=True)
                sd = df_y[ind].std(skipna=True)
                if sd is not None and sd > 0:
                    df_y[ind] = (df_y[ind] - mu) / sd
        z_panels.append(df_y)

    panel_z = pd.concat(z_panels, ignore_index=True)

    # Equal-weight DII theo năm
    panel_z["DII_equal"] = panel_z[core_inds].mean(axis=1, skipna=True)

    # Scale 0–100 theo từng năm
    def minmax(s):
        return (s - s.min()) / (s.max() - s.min()) * 100 if s.max() > s.min() else s

    panel_z["DII_equal_100"] = (
        panel_z.groupby("year")["DII_equal"].transform(minmax)
    )

    # Save outputs
    out_long = PROCESSED_DIR / "dii_panel_evolution_long.csv"
    panel_z.to_csv(out_long, index=False)

    # Country-year matrix (wide by year, optional)
    out_pivot = PROCESSED_DIR / "dii_panel_evolution_wide.csv"
    panel_z.pivot(
        index="country_iso3", columns="year", values="DII_equal_100"
    ).to_csv(out_pivot)

    print(f"Saved panel evolution: {out_long}")
    print(f"Saved panel evolution (pivot): {out_pivot}")


if __name__ == "__main__":
    main()
