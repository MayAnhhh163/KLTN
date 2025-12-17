import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import date
from dii_thesis.src.s0_settings import WDI_BASE, CONFIG_DIR, PROCESSED_DIR, DATE_RANGE, LOG_DIR

def _get_json(url: str, params: dict, retries: int = 5, backoff: float = 1.5):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json(), r.url
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")

def fetch_indicator_long(indicator_code: str, date_range: str) -> pd.DataFrame:
    """
    Fetch 1 indicator from WB:
    https://api.worldbank.org/v2/country/all/indicator/{code}?date=YYYY:YYYY&format=json&per_page=20000&page=1
    """
    url = f"{WDI_BASE}/country/all/indicator/{indicator_code}"
    params = {
        "format": "json",
        "per_page": 20000,
        "page": 1,
        "date": date_range
    }

    rows = []
    last_url = None

    while True:
        data, last_url = _get_json(url, params=params)

        # WB format: [meta, items]
        meta = data[0]
        items = data[1] if len(data) > 1 else []

        for it in items:
            rows.append({
                "country_iso3": it.get("countryiso3code"),
                "country_name": (it.get("country") or {}).get("value"),
                "year": int(it["date"]) if it.get("date") else None,
                "indicator_code": indicator_code,
                "indicator_name": (it.get("indicator") or {}).get("value"),
                "value": it.get("value"),
                "ingest_date": str(date.today()),
                "api_url": last_url
            })

        if params["page"] >= meta.get("pages", 1):
            break
        params["page"] += 1

    df = pd.DataFrame(rows)
    return df

def main():
    catalog_path = CONFIG_DIR / "indicator_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    indicators = catalog["indicator_code"].dropna().unique().tolist()

    all_dfs = []
    for code in tqdm(indicators, desc="Fetching indicators"):
        df_i = fetch_indicator_long(code, DATE_RANGE)
        all_dfs.append(df_i)

    raw_long = pd.concat(all_dfs, ignore_index=True)

    # Clean minimal
    raw_long = raw_long.dropna(subset=["country_iso3", "year"]).copy()

    # Save parquet + csv
    out_parquet = PROCESSED_DIR / "wdi_raw_long.parquet"
    out_csv = PROCESSED_DIR / "wdi_raw_long.csv"
    raw_long.to_parquet(out_parquet, index=False)
    raw_long.to_csv(out_csv, index=False)

    # Build ISO2->ISO3 mapping from data, for later join
    iso_map = raw_long[["country_name", "country_iso3"]].dropna().drop_duplicates()
    iso_map_out = PROCESSED_DIR / "country_iso3_from_wdi.csv"
    iso_map.to_csv(iso_map_out, index=False)

    print(f"Saved: {out_parquet}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {iso_map_out}")
    print(f"rows={len(raw_long)}")

if __name__ == "__main__":
    main()
