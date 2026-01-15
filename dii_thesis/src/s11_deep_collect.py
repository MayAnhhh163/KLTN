# dii_thesis/src/s11_deep_collect.py
# Patch v6: make FINDEX parser robust (no hard crash), always emit diagnostics preview,
# support both long and wide layouts; keep OECD/A4AI/ITU behavior unchanged.
# Author: ChatGPT (patch for KLTN)

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s11_deep_collect")


# -------------------------
# Utils
# -------------------------
def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_path: Path, timeout: int = 120) -> Path:
    ensure_dir(out_path.parent)
    logger.info(f"[DOWNLOAD] {url} -> {out_path}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; KLTN-DII/1.0)"}
    r = requests.get(url, stream=True, timeout=timeout, headers=headers)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return out_path


def normalize_country_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-\&\.\']", "", s)
    return s


def build_iso3_lookup_from_dii(dii_panel: pd.DataFrame) -> Dict[str, str]:
    tmp = dii_panel[["country_name", "country_iso3"]].dropna().drop_duplicates()
    mp = {}
    for _, r in tmp.iterrows():
        mp[normalize_country_name(str(r["country_name"]))] = str(r["country_iso3"])
    return mp


def detect_year_columns(cols: List[str]) -> List[str]:
    years = []
    for c in cols:
        s = str(c).strip()
        if re.fullmatch(r"(19|20)\d{2}", s):
            years.append(s)
    return years


# -------------------------
# OECD (kept minimal)
# -------------------------
def oecd_download_csv(base_url: str, agency_dataset: str, selection: str, start_period: str, end_period: str) -> pd.DataFrame:
    url = (
        f"{base_url}/data/{agency_dataset}/{selection}"
        f"?startPeriod={start_period}&endPeriod={end_period}"
        f"&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
    )
    logger.info(f"[OECD] GET {url}")
    return pd.read_csv(url)


def parse_oecd_to_long(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # common SDMX-CSV columns
    def pick(cols, candidates):
        for c in cols:
            if str(c).upper() in candidates:
                return c
        return None

    cols = list(d.columns)
    time_col = pick(cols, {"TIME_PERIOD", "TIME"})
    geo_col = pick(cols, {"REF_AREA", "LOCATION", "GEO"})
    val_col = pick(cols, {"OBS_VALUE", "VALUE"})

    if not time_col:
        for c in cols:
            if "time" in str(c).lower():
                time_col = c
                break
    if not geo_col:
        for c in cols:
            if any(k in str(c).lower() for k in ["ref_area", "location", "geo", "country"]):
                geo_col = c
                break
    if not val_col:
        for c in cols:
            if "obs" in str(c).lower() and "value" in str(c).lower():
                val_col = c
                break
        if not val_col:
            for c in cols:
                if str(c).lower() == "value":
                    val_col = c
                    break

    if not (time_col and geo_col and val_col):
        raise ValueError("OECD CSV: cannot detect TIME/REF_AREA/OBS_VALUE")

    out = d[[geo_col, time_col, val_col]].copy()
    out.columns = ["country_name", "year", "value"]
    return out


# -------------------------
# FINDEX robust parser
# -------------------------
def _guess_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for k in keys:
        for c in cols:
            if k in str(c).strip().lower():
                return c
    return None


def _try_parse_long_layout(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Try find columns for long layout: country, year, value, series/indicator name.
    """
    country_col = _guess_col(df, ["economy", "country", "location", "economies"])
    year_col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["year", "time", "time_period", "period"]:
            year_col = c
            break
    if year_col is None:
        # looser match
        year_col = _guess_col(df, ["year", "time"])

    value_col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["value", "obs_value", "val", "data", "datavalue"]:
            value_col = c
            break
    if value_col is None:
        value_col = _guess_col(df, ["value", "obs_value", "data"])

    series_col = _guess_col(df, ["indicator", "series", "question", "variable", "name", "description", "topic"])

    return country_col, year_col, value_col, series_col


def parse_findex_to_long(df: pd.DataFrame, indicator_specs: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Never raises. If cannot parse, returns empty long_df and writes preview for diagnostics.
    Supports:
      - long layout: country + year + value + (series/indicator)
      - wide layout: country + year columns + (series/indicator) optional
      - hybrid: country + many columns; keyword search in column names
    """
    d = df.copy()

    # Create a useful preview (first 40 rows, first 25 cols)
    preview_df = d.iloc[:40, : min(25, d.shape[1])].copy()

    cols = [str(c) for c in d.columns]
    year_cols = detect_year_columns(cols)

    country_col, year_col, value_col, series_col = _try_parse_long_layout(d)

    meta = {x["code"]: x for x in indicator_specs}

    long_rows = []

    # Case 1: Long layout
    if country_col and year_col and value_col:
        # Ensure year is numeric-ish
        tmp = d[[country_col, year_col, value_col] + ([series_col] if series_col else [])].copy()
        tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce").astype("Int64")
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna(subset=[year_col, value_col])

        for spec in indicator_specs:
            code = spec["code"]
            pats = spec.get("patterns", [])
            if series_col and pats:
                mask = tmp[series_col].astype(str).str.contains("|".join(pats), regex=True, na=False)
                sub = tmp[mask].copy()
            else:
                sub = tmp.copy()

            if sub.empty:
                logger.warning(f"[FINDEX] No match for {code}. See diagnostics/findex_preview_head.csv")
                continue

            out = sub[[country_col, year_col, value_col]].copy()
            out.columns = ["country_name", "year", "value"]
            out["indicator_code"] = code
            out["pillar"] = meta.get(code, {}).get("pillar", "meaningful_use")
            out["direction"] = meta.get(code, {}).get("direction", "positive")
            long_rows.append(out)

        if long_rows:
            return pd.concat(long_rows, ignore_index=True), preview_df
        return pd.DataFrame(columns=["country_name", "year", "value", "indicator_code", "pillar", "direction"]), preview_df

    # Case 2: Wide-by-year layout with year columns
    if country_col and year_cols:
        base = d[[country_col] + year_cols + ([series_col] if series_col else [])].copy()

        for spec in indicator_specs:
            code = spec["code"]
            pats = spec.get("patterns", [])
            if series_col and pats:
                sub = base[base[series_col].astype(str).str.contains("|".join(pats), regex=True, na=False)].copy()
            else:
                sub = base.copy()

            if sub.empty:
                logger.warning(f"[FINDEX] No match for {code}. See diagnostics/findex_preview_head.csv")
                continue

            melted = sub.melt(id_vars=[country_col], value_vars=year_cols, var_name="year", value_name="value")
            melted.rename(columns={country_col: "country_name"}, inplace=True)
            melted["year"] = pd.to_numeric(melted["year"], errors="coerce").astype("Int64")
            melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
            melted = melted.dropna(subset=["year", "value"])
            melted["indicator_code"] = code
            melted["pillar"] = meta.get(code, {}).get("pillar", "meaningful_use")
            melted["direction"] = meta.get(code, {}).get("direction", "positive")
            long_rows.append(melted[["country_name", "year", "value", "indicator_code", "pillar", "direction"]])

        if long_rows:
            return pd.concat(long_rows, ignore_index=True), preview_df
        return pd.DataFrame(columns=["country_name", "year", "value", "indicator_code", "pillar", "direction"]), preview_df

    # Case 3: Unknown layout -> try keyword search in column names, but do not crash
    logger.warning("[FINDEX] Cannot confidently detect layout (no country/year/value). Returning empty long. See diagnostics/findex_preview_head.csv")
    return pd.DataFrame(columns=["country_name", "year", "value", "indicator_code", "pillar", "direction"]), preview_df


# -------------------------
# A4AI: tolerant reader
# -------------------------
def load_a4ai_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    frames = []
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=s)
            df["__sheet__"] = s
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_a4ai(df: pd.DataFrame, years: List[int], value_columns_regex: str) -> pd.DataFrame:
    """
    Tries to parse the A4AI affordability sheet(s).
    Expect something like: Country + columns that contain year.
    """
    d = df.copy()
    if d.empty:
        raise ValueError("A4AI: empty dataframe")

    # Guess country col
    country_col = None
    for c in d.columns:
        lc = str(c).strip().lower()
        if lc == "country" or "country" in lc or "economy" in lc:
            country_col = c
            break
    if country_col is None:
        # fallback: first object-like column
        for c in d.columns:
            if d[c].dtype == object:
                country_col = c
                break
    if country_col is None:
        raise ValueError("A4AI: cannot detect country column")

    cols = [str(c) for c in d.columns]
    target_cols = []
    for y in years:
        # primary regex: user-provided
        pat = re.compile(value_columns_regex + r".*(" + str(y) + r")", flags=re.I)
        found = None
        for c in cols:
            if pat.search(c):
                found = c
                break
        if found is None:
            # fallback: any col containing the year
            for c in cols:
                if str(y) in c:
                    found = c
                    break
        if found:
            target_cols.append(found)

    if not target_cols:
        raise ValueError("A4AI: cannot detect year columns")

    out = d[[country_col] + target_cols].copy()
    out = out.melt(id_vars=[country_col], value_vars=target_cols, var_name="col", value_name="value")
    out.rename(columns={country_col: "country_name"}, inplace=True)
    out["year"] = out["col"].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["year", "value"])
    out = out.drop(columns=["col"])
    return out[["country_name", "year", "value"]]


# -------------------------
# ITU DataHub CSV parser (already stable in your pipeline)
# -------------------------
def parse_itu_datahub_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    # Required columns in your exports: entityIso, dataYear, dataValue
    cols = {c.lower(): c for c in d.columns}
    iso_col = cols.get("entityiso") or cols.get("iso3") or cols.get("iso") or cols.get("entity_iso")
    year_col = cols.get("datayear") or cols.get("year") or cols.get("time")
    val_col = cols.get("datavalue") or cols.get("value") or cols.get("obs_value")
    series_col = cols.get("seriescode") or cols.get("series") or cols.get("indicatorcode")

    if not (iso_col and year_col and val_col):
        raise ValueError(f"ITU: cannot detect required cols in {path.name}")

    out = d[[iso_col, year_col, val_col] + ([series_col] if series_col else [])].copy()
    out.rename(columns={iso_col: "country_iso3", year_col: "year", val_col: "value"}, inplace=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if series_col:
        out.rename(columns={series_col: "seriesCode"}, inplace=True)
    else:
        out["seriesCode"] = None
    out = out.dropna(subset=["country_iso3", "year", "value"])
    return out


# -------------------------
# Main
# -------------------------
def _get_out_dir(cfg: dict) -> Path:
    p = cfg.get("paths", {})
    for k in ["out_dir", "outdir", "output_dir", "out_dir_path"]:
        if k in p:
            return Path(p[k])
    # fallback default
    return Path("dii_thesis/data/processed/dii_deep")


def main(config_path: str) -> None:
    cfg = read_yaml(Path(config_path))

    out_dir = _get_out_dir(cfg)
    external_dir = Path(cfg.get("paths", {}).get("external_dir", "dii_thesis/data/external"))
    dii_panel_path = Path(cfg.get("paths", {}).get("dii_core_panel", "dii_thesis/data/processed/dii_core/dii_core_panel.csv"))

    ensure_dir(out_dir / "raw_long")
    diagnostics_dir = out_dir / "diagnostics"
    ensure_dir(diagnostics_dir)
    ensure_dir(external_dir)

    dii_panel = pd.read_csv(dii_panel_path)
    iso3_lookup = build_iso3_lookup_from_dii(dii_panel)

    long_all = []

    # OECD (optional)
    if cfg.get("sources", {}).get("oecd", {}).get("enabled", False):
        base_url = cfg.get("collect", {}).get("oecd_base_url", "https://sdmx.oecd.org/public/rest")
        for req in cfg["sources"]["oecd"].get("requests", []):
            try:
                raw = oecd_download_csv(
                    base_url=base_url,
                    agency_dataset=req["agency_dataset"],
                    selection=req["selection"],
                    start_period=req["startPeriod"],
                    end_period=req["endPeriod"],
                )
                long = parse_oecd_to_long(raw)
                long["indicator_code"] = req["out_indicator_code"]
                long["pillar"] = req["pillar"]
                long["direction"] = req["direction"]
                long_all.append(long)
            except Exception as e:
                logger.warning(f"[OECD] That bai {req.get('out_indicator_code')} | {e}")

    # FINDEX
    if cfg.get("sources", {}).get("findex", {}).get("enabled", False):
        findex_dir = external_dir / "findex"
        ensure_dir(findex_dir)

        # unify file name
        csv_path = findex_dir / "GlobalFindexDatabase.csv"
        if not csv_path.exists():
            url = cfg.get("collect", {}).get("findex_csv_url")
            if not url:
                raise ValueError("FINDEX enabled but collect.findex_csv_url is missing")
            download_file(url, csv_path)

        df_fx = pd.read_csv(csv_path, low_memory=False)
        fx_long, fx_preview = parse_findex_to_long(df_fx, cfg["sources"]["findex"].get("indicators", []))
        fx_preview.to_csv(diagnostics_dir / "findex_preview_head.csv", index=False)

        # If parsed empty, continue without crashing
        if not fx_long.empty:
            long_all.append(fx_long)

    # A4AI
    if cfg.get("sources", {}).get("a4ai", {}).get("enabled", False):
        a4ai_dir = external_dir / "a4ai"
        ensure_dir(a4ai_dir)
        xlsx_path = a4ai_dir / "AffordabilityData_2015-2017.xlsx"
        if not xlsx_path.exists():
            url = cfg.get("collect", {}).get("a4ai_mobile_pricing_xlsx_url")
            if not url:
                raise ValueError("A4AI enabled but collect.a4ai_mobile_pricing_xlsx_url missing")
            download_file(url, xlsx_path)

        df_a4 = load_a4ai_xlsx(xlsx_path)
        for spec in cfg["sources"]["a4ai"].get("indicators", []):
            try:
                long = parse_a4ai(
                    df_a4,
                    years=spec.get("years", [2015, 2016, 2017]),
                    value_columns_regex=spec.get("value_columns_regex", r"PRICE AS % OF INCOME"),
                )
                long["indicator_code"] = spec["code"]
                long["pillar"] = spec["pillar"]
                long["direction"] = spec["direction"]
                long_all.append(long)
            except Exception as e:
                logger.warning(f"[A4AI] Failed {spec.get('code')} | {e}")

    # ITU
    if cfg.get("sources", {}).get("itu", {}).get("enabled", False):
        itu_dir = external_dir / "itu"
        ensure_dir(itu_dir)

        for spec in cfg["sources"]["itu"].get("indicators", []):
            file_glob = spec.get("file_glob")
            if not file_glob:
                continue
            files = list(itu_dir.glob(file_glob))
            if not files:
                logger.warning(f"[ITU] Chua co file cho {spec.get('code')} ({file_glob})")
                continue

            # Combine all matched files
            frames = []
            for fp in files:
                try:
                    t = parse_itu_datahub_csv(fp)
                    t["__file__"] = fp.name
                    frames.append(t)
                except Exception as e:
                    logger.warning(f"[ITU] Doc file that bai: {fp.name} | {e}")
            if not frames:
                continue

            df_itu = pd.concat(frames, ignore_index=True)

            # Optional: filter seriesCode
            series_filter = spec.get("series_filter_regex")
            if series_filter and "seriesCode" in df_itu.columns:
                df_itu = df_itu[df_itu["seriesCode"].astype(str).str.contains(series_filter, regex=True, na=False)]

            # Mode: keep seriescode (explode) OR aggregate
            keep_seriescode = bool(spec.get("keep_seriescode", False))
            aggregate_series = bool(spec.get("aggregate_series", False))

            if keep_seriescode and "seriesCode" in df_itu.columns:
                # Each seriesCode becomes an indicator_code suffix
                df_itu["indicator_code"] = df_itu["seriesCode"].astype(str).map(lambda x: f"{spec['code']}__{x}")
                out = df_itu[["country_iso3", "year", "value", "indicator_code"]].copy()
                out["pillar"] = spec["pillar"]
                out["direction"] = spec["direction"]
                # Map iso3 to country_name using DII panel (if available)
                iso_to_name = dii_panel[["country_iso3", "country_name"]].drop_duplicates().set_index("country_iso3")["country_name"].to_dict()
                out["country_name"] = out["country_iso3"].map(iso_to_name)
                long_all.append(out[["country_name", "year", "value", "indicator_code", "pillar", "direction"]])
            else:
                if aggregate_series and "seriesCode" in df_itu.columns:
                    # Average across series (for umbrella indicators)
                    out = df_itu.groupby(["country_iso3", "year"], as_index=False)["value"].mean()
                else:
                    out = df_itu[["country_iso3", "year", "value"]].copy()

                out["indicator_code"] = spec["code"]
                out["pillar"] = spec["pillar"]
                out["direction"] = spec["direction"]
                iso_to_name = dii_panel[["country_iso3", "country_name"]].drop_duplicates().set_index("country_iso3")["country_name"].to_dict()
                out["country_name"] = out["country_iso3"].map(iso_to_name)
                long_all.append(out[["country_name", "year", "value", "indicator_code", "pillar", "direction"]])

    if not long_all:
        raise RuntimeError("Khong thu thap duoc bat ky du lieu deep nao. Hay kiem tra config/nguon du lieu.")

    deep_long = pd.concat(long_all, ignore_index=True)

    # Normalize, map iso3 by DII country list if missing
    deep_long["country_name_raw"] = deep_long["country_name"].astype(str)
    deep_long["country_name_norm"] = deep_long["country_name_raw"].map(normalize_country_name)
    deep_long["country_iso3"] = deep_long.get("country_iso3") if "country_iso3" in deep_long.columns else None
    if "country_iso3" not in deep_long.columns or deep_long["country_iso3"].isna().all():
        deep_long["country_iso3"] = deep_long["country_name_norm"].map(iso3_lookup)

    deep_long["year"] = pd.to_numeric(deep_long["year"], errors="coerce").astype("Int64")
    deep_long["value"] = pd.to_numeric(deep_long["value"], errors="coerce")
    deep_long = deep_long.dropna(subset=["country_iso3", "year", "value", "indicator_code"])

    # Apply time filter if provided
    tcfg = cfg.get("time", {})
    start_year = int(tcfg.get("start_year", 2015))
    end_year = int(tcfg.get("end_year", 2025))
    deep_long = deep_long[(deep_long["year"] >= start_year) & (deep_long["year"] <= end_year)].copy()

    # Save
    out_path = out_dir / "raw_long" / "dii_deep_raw_long.csv"
    deep_long[["country_iso3","country_name","year","value","indicator_code","pillar","direction"]].to_csv(out_path, index=False)

    logger.info(
        f"[OK] Saved: {out_path} | rows={len(deep_long):,} | indicators={deep_long['indicator_code'].nunique()} | countries={deep_long['country_iso3'].nunique()}"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
