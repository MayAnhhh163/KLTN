import pandas as pd
from dii_thesis.src.s0_settings import PROCESSED_DIR, CONFIG_DIR, EXCLUDE_REGION_VALUE

def main():
    raw_long = pd.read_parquet(PROCESSED_DIR / "wdi_raw_long.parquet")
    catalog = pd.read_csv(CONFIG_DIR / "indicator_catalog.csv")

    # Pivot
    wide = raw_long.pivot_table(
        index=["country_iso3", "year"],
        columns="indicator_code",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Load country metadata (iso2-based) and attempt to enrich using name
    country_pre = pd.read_csv(PROCESSED_DIR / "country_dim_pre_iso3.csv")
    iso3_from_wdi = pd.read_csv(PROCESSED_DIR / "country_iso3_from_wdi.csv")

    # Join iso3 into country_pre via country_name match (best-effort)
    # Thực tế: WB naming khá ổn, nhưng vẫn có mismatch nhỏ. Đây là lý do ta ưu tiên ISO3 từ indicator data.
    country_dim = country_pre.merge(
        iso3_from_wdi,
        left_on="country_name",
        right_on="country_name",
        how="left"
    )

    # Keep only rows with ISO3 (country-level)
    country_dim = country_dim.dropna(subset=["country_iso3"]).copy()

    # Remove aggregates again (safety)
    country_dim = country_dim[country_dim["region"].fillna("") != EXCLUDE_REGION_VALUE].copy()

    # Deduplicate ISO3
    country_dim = country_dim.sort_values(["country_iso3"]).drop_duplicates("country_iso3")

    # Join region/income into wide
    wide = wide.merge(
        country_dim[["country_iso3", "country_name", "region", "income_group", "lending_type"]],
        on="country_iso3",
        how="left"
    )

    out_country = PROCESSED_DIR / "country_dim.csv"
    out_wide = PROCESSED_DIR / "dii_panel_wide.csv"

    country_dim.to_csv(out_country, index=False)
    wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_country} | rows={len(country_dim)}")
    print(f"Saved: {out_wide} | rows={len(wide)}")

if __name__ == "__main__":
    main()
