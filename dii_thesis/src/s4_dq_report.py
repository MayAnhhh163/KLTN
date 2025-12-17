import pandas as pd
from dii_thesis.src.s0_settings import PROCESSED_DIR, CONFIG_DIR, DQ_WINDOW

def main():
    wide = pd.read_csv(PROCESSED_DIR / "dii_panel_wide.csv")
    catalog = pd.read_csv(CONFIG_DIR / "indicator_catalog.csv")

    y0, y1 = DQ_WINDOW
    base = wide[(wide["year"] >= y0) & (wide["year"] <= y1)].copy()

    # Total countries in base window (with ISO3 present)
    n_total = base["country_iso3"].nunique()

    core = catalog[catalog["priority"] == 1]["indicator_code"].tolist()

    rows = []
    for ind in core:
        if ind not in base.columns:
            rows.append({
                "indicator_code": ind,
                "year_from": y0,
                "year_to": y1,
                "n_countries_total": int(n_total),
                "n_countries_non_missing": 0,
                "coverage_rate": 0.0,
                "notes": "Indicator column not found after pivot"
            })
            continue

        # Country considered covered if it has at least one non-missing observation in the window
        covered = base.groupby("country_iso3")[ind].apply(lambda s: s.notna().any()).sum()
        rows.append({
            "indicator_code": ind,
            "year_from": y0,
            "year_to": y1,
            "n_countries_total": int(n_total),
            "n_countries_non_missing": int(covered),
            "coverage_rate": float(covered / n_total) if n_total else None,
            "notes": ""
        })

    dq = pd.DataFrame(rows).sort_values("coverage_rate", ascending=True)
    out = PROCESSED_DIR / "dq_report.csv"
    dq.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(dq)

if __name__ == "__main__":
    main()
