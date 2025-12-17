import time
import requests
import pandas as pd
from dii_thesis.src.s0_settings import WDI_BASE, PROCESSED_DIR, EXCLUDE_REGION_VALUE

def _get_json(url: str, params: dict, retries: int = 5, backoff: float = 1.5):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")

def fetch_country_dim(per_page: int = 400) -> pd.DataFrame:
    """
    Tải danh sách country metadata từ World Bank API:
    - Lọc bỏ aggregates (region.value == 'Aggregates')
    - Chỉ giữ entity có ISO3 hợp lệ (len==3)
    """
    url = f"{WDI_BASE}/country"
    params = {"format": "json", "per_page": per_page, "page": 1}

    rows = []
    while True:
        data = _get_json(url, params=params)
        meta, items = data[0], data[1]
        for it in items:
            iso3 = it.get("id")  # WB uses 'id' as country code (often ISO2), but also has 'iso2Code'
            iso2 = it.get("iso2Code")
            iso3_alt = it.get("id")  # not reliable for ISO3

            # WB country endpoint returns: id (2-letter) and may include additional fields; ISO3 is not here.
            # Reliable ISO3 comes from WDI series responses via countryiso3code.
            # Tuy nhiên: vẫn lưu iso2 + name + region + income; ISO3 sẽ “fill” sau bằng mapping từ indicator fetch.
            rows.append({
                "country_id": it.get("id"),
                "iso2": iso2,
                "country_name": it.get("name"),
                "region": (it.get("region") or {}).get("value"),
                "income_group": (it.get("incomeLevel") or {}).get("value"),
                "lending_type": (it.get("lendingType") or {}).get("value"),
                "capital_city": it.get("capitalCity"),
                "longitude": it.get("longitude"),
                "latitude": it.get("latitude"),
            })

        if params["page"] >= meta.get("pages", 1):
            break
        params["page"] += 1

    df = pd.DataFrame(rows)

    # Lọc aggregates
    df = df[df["region"].fillna("") != EXCLUDE_REGION_VALUE].copy()

    # Một số entity không có income/region rõ, vẫn giữ, sẽ lọc sau khi có ISO3 mapping
    return df

def main():
    df = fetch_country_dim()
    out = PROCESSED_DIR / "country_dim_pre_iso3.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out} | rows={len(df)}")

if __name__ == "__main__":
    main()
