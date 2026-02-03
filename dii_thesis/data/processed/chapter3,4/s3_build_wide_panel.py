import pandas as pd

from dii_thesis.src.s0_settings import (
    PROCESSED_DIR, CONFIG_DIR, EXCLUDE_REGION_VALUE, DQ_WINDOW
)


def build_country_dim() -> pd.DataFrame:
    """
    Build country_dim (đã có ISO3) bằng cách:
    - đọc country_dim_pre_iso3.csv (metadata từ /country, đã lọc Aggregates theo region)
    - đọc country_iso3_from_wdi.csv (mapping country_name -> ISO3 từ indicator endpoint)
    - merge theo country_name
    - lọc ISO3 hợp lệ, loại Aggregates lần nữa (defensive)
    - dedup theo ISO3
    """
    country_pre = pd.read_csv(PROCESSED_DIR / "country_dim_pre_iso3.csv")
    iso3_from_wdi = pd.read_csv(PROCESSED_DIR / "country_iso3_from_wdi.csv")

    # Chuẩn hoá tên để tăng tỷ lệ match
    country_pre["country_name_key"] = country_pre["country_name"].astype(str).str.strip().str.lower()
    iso3_from_wdi["country_name_key"] = iso3_from_wdi["country_name"].astype(str).str.strip().str.lower()

    country_dim = country_pre.merge(
        iso3_from_wdi[["country_name_key", "country_iso3"]],
        on="country_name_key",
        how="left"
    )

    # dọn cột key phụ
    country_dim = country_dim.drop(columns=["country_name_key"], errors="ignore")

    # clean ISO3
    country_dim["country_iso3"] = country_dim["country_iso3"].astype("string").str.strip()
    country_dim = country_dim.dropna(subset=["country_iso3"]).copy()
    country_dim = country_dim[country_dim["country_iso3"].str.len() == 3].copy()

    # loại aggregates lần nữa (an toàn)
    country_dim = country_dim[country_dim["region"].fillna("") != EXCLUDE_REGION_VALUE].copy()

    # dedup theo ISO3
    country_dim = (
        country_dim
        .sort_values(["country_iso3"])
        .drop_duplicates(subset=["country_iso3"], keep="first")
        .reset_index(drop=True)
    )

    return country_dim


def compute_dq_report_country_only(
    raw_long: pd.DataFrame,
    country_whitelist: set,
    catalog: pd.DataFrame,
    dq_window: tuple[int, int]
) -> pd.DataFrame:
    """
    Tạo dq_report theo country-only, trong cửa sổ dq_window (ví dụ 2018-2022):
    - n_countries_total: số quốc gia mục tiêu (len(country_whitelist))
    - n_countries_with_data: số quốc gia có ít nhất 1 value != null trong window cho indicator đó
    - coverage_rate: n_with_data / n_total
    - n_obs_total: số quan sát (rows) trong window cho indicator (country-year)
    - missing_rate: tỷ lệ missing theo (country-year) trong window trên tập country_whitelist
      missing_rate = 1 - (n_non_null / (n_total_countries * n_years))
    """
    start_y, end_y = dq_window
    years = list(range(start_y, end_y + 1))
    n_years = len(years)

    # chỉ giữ country-only + window
    df = raw_long.copy()
    df["country_iso3"] = df["country_iso3"].astype("string").str.strip()
    df = df[df["country_iso3"].isin(country_whitelist)].copy()
    df = df[df["year"].isin(years)].copy()

    if "indicator_code" not in catalog.columns:
        raise ValueError("indicator_catalog.csv phải có cột 'indicator_code'")

    ind_list = (
        catalog["indicator_code"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    n_total = len(country_whitelist)
    denom_country_year = n_total * n_years if n_total > 0 else 0

    rows = []
    for code in ind_list:
        df_i = df[df["indicator_code"].astype(str).str.strip() == code].copy()

        # số quan sát (country-year) có record
        n_obs_total = len(df_i)

        # số non-null value
        n_non_null = df_i["value"].notna().sum()

        # số quốc gia có ít nhất 1 non-null trong window
        n_countries_with_data = (
            df_i[df_i["value"].notna()][["country_iso3"]]
            .drop_duplicates()
            .shape[0]
        )

        coverage_rate = (n_countries_with_data / n_total) if n_total > 0 else 0.0
        missing_rate = (1 - (n_non_null / denom_country_year)) if denom_country_year > 0 else 1.0

        rows.append({
            "indicator_code": code,
            "dq_window_start": start_y,
            "dq_window_end": end_y,
            "n_countries_total": n_total,
            "n_countries_with_data": n_countries_with_data,
            "coverage_rate": coverage_rate,
            "n_obs_total": n_obs_total,
            "missing_rate_country_year": missing_rate,
        })

    return pd.DataFrame(rows).sort_values(["coverage_rate", "missing_rate_country_year"], ascending=[False, True])


def main():
    # 1) load raw_long
    raw_long = pd.read_parquet(PROCESSED_DIR / "wdi_raw_long.parquet")
    raw_long["country_iso3"] = raw_long["country_iso3"].astype("string").str.strip()

    # 2) build country_dim + whitelist ISO3 (country-only)
    country_dim = build_country_dim()
    whitelist = set(country_dim["country_iso3"].dropna().astype(str).str.strip().tolist())

    # 3) lọc raw_long theo whitelist (đây là bước quan trọng để loại Aggregates triệt để)
    raw_long_clean = raw_long[raw_long["country_iso3"].isin(whitelist)].copy()

    # clean year
    raw_long_clean = raw_long_clean.dropna(subset=["year"]).copy()
    raw_long_clean["year"] = raw_long_clean["year"].astype(int)

    # loại trùng chắc chắn lần nữa
    raw_long_clean = (
        raw_long_clean
        .sort_values(["country_iso3", "year", "indicator_code"])
        .drop_duplicates(subset=["country_iso3", "year", "indicator_code"], keep="first")
        .reset_index(drop=True)
    )

    # 4) pivot long -> wide
    wide = (
        raw_long_clean
        .pivot_table(
            index=["country_iso3", "year"],
            columns="indicator_code",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # 5) enrich metadata vào panel
    wide = wide.merge(
        country_dim[["country_iso3", "country_name", "region", "income_group", "lending_type"]],
        on="country_iso3",
        how="left"
    )

    # 6) đảm bảo panel sạch: không ISO3 NaN, không year NaN, chỉ whitelist
    wide = wide.dropna(subset=["country_iso3", "year"]).copy()
    wide = wide[wide["country_iso3"].isin(whitelist)].copy()
    wide["year"] = wide["year"].astype(int)

    # 7) save outputs
    out_country = PROCESSED_DIR / "country_dim.csv"
    out_wide = PROCESSED_DIR / "dii_panel_wide.csv"

    country_dim.to_csv(out_country, index=False)
    wide.to_csv(out_wide, index=False)

    # 8) build dq_report (country-only)
    catalog = pd.read_csv(CONFIG_DIR / "indicator_catalog.csv")
    dq_report = compute_dq_report_country_only(
        raw_long=raw_long_clean,
        country_whitelist=whitelist,
        catalog=catalog,
        dq_window=DQ_WINDOW
    )
    out_dq = PROCESSED_DIR / "dq_report.csv"
    dq_report.to_csv(out_dq, index=False)

    # 9) log summary
    print(f"Saved: {out_country} | rows={len(country_dim)} | countries={len(whitelist)}")
    print(f"Saved: {out_wide} | rows={len(wide)} | panel_countries={wide['country_iso3'].nunique()} | years={wide['year'].nunique()}")
    print(f"Saved: {out_dq} | rows={len(dq_report)} | window={DQ_WINDOW[0]}-{DQ_WINDOW[1]}")

    # sanity checks
    n_bad_iso3 = wide["country_iso3"].isna().sum()
    if n_bad_iso3 > 0:
        print(f"Warning: panel vẫn còn {n_bad_iso3} dòng thiếu country_iso3 (không nên xảy ra).")

    # check aggregates leakage: iso3 trong panel nhưng không có trong whitelist
    leakage = set(wide["country_iso3"].dropna().astype(str)) - whitelist
    if len(leakage) > 0:
        print(f"Warning: phát hiện leakage ISO3 ngoài whitelist: {sorted(list(leakage))[:20]} (show first 20)")


if __name__ == "__main__":
    main()
