from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

# -----------------------
# URLs (dùng trong code được)
# -----------------------
URL_EGDI = "https://data360files.worldbank.org/data360-data/data/UN_EGDI/UN_EGDI_WIDEF.csv"
URL_NRI_2022 = "https://download.networkreadinessindex.org/reports/data/2022/nri-2022-dataset.xlsx"
URL_MCI_2022 = "https://www.mobileconnectivityindex.com/widgets/connectivityIndex/excel/MCI_Data_2022.xlsx"

# -----------------------
# Helpers
# -----------------------
def _clean_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _require_cols(df: pd.DataFrame, cols: list[str], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc {where}: {missing}")

def spearman(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 10:
        return float("nan")
    return float(tmp["a"].corr(tmp["b"], method="spearman"))

def pearson(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 10:
        return float("nan")
    return float(tmp["a"].corr(tmp["b"], method="pearson"))

def dense_rank_desc(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="dense")

# -----------------------
# Load DII
def load_dii_panel(panel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(panel_path)

    needed = ["country_iso3", "country_name", "year", "dii_core_0_100"]
    _require_cols(df, needed, where="DII panel")

    # giữ thêm meta nếu có
    keep = needed.copy()
    optional = ["income_group", "region", "lending_type",
                # GDPpc candidates (bạn giữ cái nào có trong panel thực tế)
                "gdp_pc", "gdp_per_capita",
                "NY.GDP.PCAP.CD", "NY.GDP.PCAP.KD", "NY.GDP.PCAP.PP.KD"]
    for c in optional:
        if c in df.columns:
            keep.append(c)

    df = df[keep].copy()
    df["year"] = df["year"].astype(int)
    return df

def dii_country_year(df_panel: pd.DataFrame, year: int) -> pd.DataFrame:
    d = df_panel[df_panel["year"] == year].copy()

    keep = ["country_iso3", "country_name", "year", "dii_core_0_100"]
    for c in ["income_group", "region", "lending_type",
              "gdp_pc", "gdp_per_capita",
              "NY.GDP.PCAP.CD", "NY.GDP.PCAP.KD", "NY.GDP.PCAP.PP.KD"]:
        if c in d.columns:
            keep.append(c)

    out = d[keep].copy()
    out = out.rename(columns={"dii_core_0_100": "dii_0_100"})
    return out


def dii_country_period_mean(df_panel: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    d = df_panel[(df_panel["year"] >= start) & (df_panel["year"] <= end)].copy()
    g = (
        d.groupby(["country_iso3", "country_name"], as_index=False)["dii_core_0_100"]
        .mean()
        .rename(columns={"dii_core_0_100": f"dii_0_100_mean_{start}_{end}"})
    )
    return g

# -----------------------
# EGDI (UN via World Bank Data360 file)
# -----------------------
def load_egdi(csv_path_or_url: str) -> pd.DataFrame:
    """
    Robust loader cho UN_EGDI_WIDEF.csv (Data360).
    Hỗ trợ 2 dạng:
    - SDMX/long: có TIME_PERIOD / REF_AREA / INDICATOR / OBS_VALUE ...
    - wide: có các cột năm 4 chữ số
    Trả về: country_iso3, year, egdi_score (+ optional country_name_egdi)
    """
    df = pd.read_csv(csv_path_or_url)

    def pick_col(candidates: list[str]) -> str | None:
        cols = {str(c).strip().lower(): c for c in df.columns}
        for key in candidates:
            if key in cols:
                return cols[key]
        return None

    def pick_col_contains(substrs: list[str]) -> str | None:
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(s in cl for s in substrs):
                return c
        return None

    # 1) Thử nhận diện wide (cột năm là '2015','2016',...)
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]
    if year_cols:
        iso_col = pick_col(["iso3", "country code", "economy code", "id", "ref_area"]) or pick_col_contains(["iso"])
        name_col = pick_col(["country name", "economy", "name", "country", "ref_area_label"])
        if iso_col is None:
            raise ValueError("EGDI wide: không tìm thấy cột ISO3/REF_AREA.")

        melted = df[[iso_col] + ([name_col] if name_col else []) + year_cols].melt(
            id_vars=[iso_col] + ([name_col] if name_col else []),
            value_vars=year_cols,
            var_name="year",
            value_name="egdi_score",
        )
        melted = melted.rename(columns={iso_col: "country_iso3"})
        if name_col:
            melted = melted.rename(columns={name_col: "country_name_egdi"})
        melted["year"] = melted["year"].astype(int)
        melted["egdi_score"] = pd.to_numeric(melted["egdi_score"], errors="coerce")
        return melted

    # 2) Nhận diện SDMX/long
    year_col = pick_col(["year", "time_period"]) or pick_col_contains(["time_period", "year"])
    iso_col = pick_col(["ref_area", "iso3", "id", "country code"]) or pick_col_contains(["ref_area", "iso"])
    name_col = pick_col(["ref_area_label", "country_name", "country name", "name", "economy", "country"])
    value_col = pick_col(["obs_value", "value", "score"]) or pick_col_contains(["obs_value", "value", "score"])
    indicator_col = pick_col(["indicator"]) or pick_col_contains(["indicator"])
    indicator_label_col = pick_col(["indicator_label"]) or pick_col_contains(["indicator_label", "indicator name"])

    if year_col is None or iso_col is None or value_col is None:
        raise ValueError(
            "Không nhận diện được cấu trúc EGDI SDMX/long (thiếu year/iso/value). "
            f"Columns hiện có: {list(df.columns)}"
        )

    out = df[[iso_col, year_col, value_col] + ([name_col] if name_col else []) +
             ([indicator_col] if indicator_col else []) + ([indicator_label_col] if indicator_label_col else [])].copy()

    out = out.rename(columns={iso_col: "country_iso3", year_col: "year", value_col: "egdi_score"})
    if name_col:
        out = out.rename(columns={name_col: "country_name_egdi"})

    # Lọc đúng indicator EGDI (vì file thường có EGDI/OSI/TII/HCI)
    # Ưu tiên exact code: 'UN_EGDI_EGDI' nếu có
    if indicator_col and "country_iso3" in out.columns:
        ind = out[indicator_col].astype(str)
        if (ind == "UN_EGDI_EGDI").any():
            out = out[ind == "UN_EGDI_EGDI"].copy()
        else:
            # fallback: lấy những dòng có chứa 'EGDI' nhưng loại OSI/TII/HCI nếu dính
            out = out[ind.str.contains("EGDI", case=False, na=False)].copy()
            out = out[~ind.str.contains(r"\b(OSI|TII|HCI)\b", case=False, na=False)].copy()

    # fallback thêm theo label nếu code không ổn
    if (indicator_col is None) and indicator_label_col:
        lab = out[indicator_label_col].astype(str)
        out = out[lab.str.contains("E-Government Development Index", case=False, na=False)].copy()

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["egdi_score"] = pd.to_numeric(out["egdi_score"], errors="coerce")

    out = out.dropna(subset=["country_iso3", "year"])
    out["year"] = out["year"].astype(int)

    return out[["country_iso3"] + (["country_name_egdi"] if "country_name_egdi" in out.columns else []) + ["year", "egdi_score"]]


# -----------------------
# NRI 2022
# -----------------------
def load_nri_2022(xlsx_path_or_url: str) -> pd.DataFrame:
    """
    Robust loader cho NRI 2022 (xlsx) khi file có merged header => nhiều cột Unnamed.
    Không dựa vào tên cột; tự dò cột ISO3/Country/Score theo nội dung.

    Output: country_iso3, country_name_nri (nếu có), nri_score, (nri_rank nếu dò được), year=2022
    """
    import numpy as np
    import pandas as pd
    import re

    # đọc sheet "NRI 2022" nếu có, không thì đọc sheet đầu tiên
    xl = pd.ExcelFile(xlsx_path_or_url)
    sheet = "NRI 2022" if "NRI 2022" in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(sheet)

    # bỏ các dòng hoàn toàn trống
    df = df.dropna(how="all").copy()

    # helper: chuẩn hoá chuỗi
    def _s(x):
        return str(x).strip()

    # 1) dò ISO3 theo pattern 3 chữ in hoa
    iso_col = None
    best_iso_rate = 0.0
    for c in df.columns:
        s = df[c].dropna().astype(str).str.strip().str.upper()
        if len(s) < 20:
            continue
        rate = s.str.fullmatch(r"[A-Z]{3}").mean()
        if rate > best_iso_rate and rate >= 0.50:
            best_iso_rate = rate
            iso_col = c

    # 2) dò Score: cột numeric với phần lớn giá trị nằm trong [0,100]
    score_col = None
    best_score_rate = 0.0
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() < 0.50:
            continue
        # NRI score thường 0-100; cho phép biên rộng một chút
        in_range = num.dropna().between(-5, 105).mean() if num.notna().any() else 0.0
        if in_range > best_score_rate and in_range >= 0.80:
            best_score_rate = in_range
            score_col = c

    # ưu tiên nếu có cột tên đúng "NRI" (trường hợp của bạn)
    if "NRI" in df.columns:
        num = pd.to_numeric(df["NRI"], errors="coerce")
        if num.notna().mean() >= 0.50:
            score_col = "NRI"

    if score_col is None:
        raise ValueError(
            f"Không dò được cột NRI score theo nội dung. Sheet={sheet}. "
            f"Gợi ý: hãy mở file và kiểm tra cột score nằm trong khoảng 0–100."
        )

    # 3) dò Country/name: cột text (không phải ISO), có độ dài trung bình khá
    name_col = None
    best_text_len = 0.0
    for c in df.columns:
        if c == iso_col or c == score_col:
            continue
        s = df[c].dropna().astype(str).map(_s)
        if len(s) < 20:
            continue
        # loại các cột toàn số
        as_num = pd.to_numeric(s, errors="coerce")
        if as_num.notna().mean() > 0.50:
            continue
        avg_len = s.map(len).mean()
        if avg_len > best_text_len and avg_len >= 6:
            best_text_len = avg_len
            name_col = c

    # 4) dò Rank (optional): cột numeric gần như integer, nằm trong [1, 250]
    rank_col = None
    best_rank_rate = 0.0
    for c in df.columns:
        if c in (iso_col, score_col, name_col):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() < 0.50:
            continue
        in_range = num.dropna().between(1, 250).mean() if num.notna().any() else 0.0
        is_intish = (np.abs(num.dropna() - np.round(num.dropna())) < 1e-6).mean() if num.notna().any() else 0.0
        rate = in_range * is_intish
        if rate > best_rank_rate and rate >= 0.60:
            best_rank_rate = rate
            rank_col = c

    keep = [score_col]
    if iso_col is not None:
        keep.insert(0, iso_col)
    if name_col is not None:
        keep.insert(0 if iso_col is None else 1, name_col)
    if rank_col is not None:
        keep.append(rank_col)

    out = df[keep].copy()

    # rename
    rename_map = {score_col: "nri_score"}
    if iso_col is not None:
        rename_map[iso_col] = "country_iso3"
    if name_col is not None:
        rename_map[name_col] = "country_name_nri"
    if rank_col is not None:
        rename_map[rank_col] = "nri_rank"
    out = out.rename(columns=rename_map)

    # clean
    out["nri_score"] = pd.to_numeric(out["nri_score"], errors="coerce")
    if "country_iso3" in out.columns:
        out["country_iso3"] = out["country_iso3"].astype(str).str.strip().str.upper()
        out.loc[~out["country_iso3"].str.fullmatch(r"[A-Z]{3}"), "country_iso3"] = np.nan

    out["year"] = 2022

    # drop dòng không có score hoặc không có country (iso hoặc name)
    if "country_iso3" in out.columns and "country_name_nri" in out.columns:
        out = out.dropna(subset=["nri_score"]).copy()
        out = out[~(out["country_iso3"].isna() & out["country_name_nri"].isna())].copy()
    elif "country_iso3" in out.columns:
        out = out.dropna(subset=["country_iso3", "nri_score"]).copy()
    elif "country_name_nri" in out.columns:
        out = out.dropna(subset=["country_name_nri", "nri_score"]).copy()
    else:
        raise ValueError("NRI dataset không có đủ thông tin country để merge (thiếu ISO3 và name).")

    return out

# -----------------------
# MCI (file 2022 thường chứa series đến 2021)
# -----------------------
def load_mci(xlsx_path_or_url: str) -> pd.DataFrame:
    """
    Robust loader cho GSMA Mobile Connectivity Index (MCI) xlsx.
    Hỗ trợ:
      - header gộp / Unnamed -> tự dò dòng header
      - dạng wide (nhiều cột năm)
      - dạng long (có cột year)
    Quan trọng: ISO3 có thể KHÔNG có trong file => output vẫn trả về được theo country_name_mci.

    Output: (country_iso3 nếu có) + country_name_mci (nếu có) + year + mci_score
    """
    xl = pd.ExcelFile(xlsx_path_or_url)

    def norm(x) -> str:
        return re.sub(r"\s+", " ", str(x).strip().lower())

    def auto_header_parse(sheet_name: str) -> pd.DataFrame:
        raw = xl.parse(sheet_name, header=None)
        raw = raw.dropna(how="all")
        if len(raw) == 0:
            return xl.parse(sheet_name)

        best_r = 0
        best_score = -1

        for r in range(min(40, len(raw))):
            row = raw.iloc[r].astype(str).map(lambda s: s.strip())
            joined = " ".join(row.tolist()).lower()

            score = 0
            score += 2 if ("iso" in joined or "iso3" in joined) else 0
            score += 2 if ("country" in joined or "economy" in joined or "name" in joined) else 0
            score += 2 if ("year" in joined or "time" in joined) else 0
            score += 3 if re.search(r"\b20\d{2}\b", joined) else 0
            score += 1 if ("mci" in joined or "mobile connectivity" in joined) else 0
            score += 1 if ("index" in joined or "score" in joined) else 0

            if score > best_score:
                best_score = score
                best_r = r

        header = raw.iloc[best_r].tolist()
        df = raw.iloc[best_r + 1 :].copy()
        df.columns = header
        df = df.dropna(how="all")

        # loại các cột header rỗng/nan
        df = df.loc[:, [c for c in df.columns if str(c).strip().lower() not in ("nan", "none", "")]]
        df.columns = make_unique_columns(df.columns)

        return df

    def make_unique_columns(cols):
        seen = {}
        out = []
        for c in cols:
            c0 = str(c).strip()
            if c0 == "" or c0.lower() in ("nan", "none"):
                c0 = "unnamed"
            key = c0
            if key not in seen:
                seen[key] = 0
                out.append(key)
            else:
                seen[key] += 1
                out.append(f"{key}__{seen[key]}")
        return out

    def detect_iso_col(df: pd.DataFrame) -> str | None:
        best = None
        best_rate = 0.0
        for c in df.columns:
            col_data = df[c]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            s = col_data.dropna().astype(str).str.strip().str.upper()
            if len(s) < 20:
                continue
            rate = s.str.fullmatch(r"[A-Z]{3}").mean()
            if rate > best_rate and rate >= 0.30:
                best_rate = rate
                best = c
        if best is None:
            for c in df.columns:
                if "iso" in norm(c):
                    return c
        return best

    def detect_year_col(df: pd.DataFrame) -> str | None:
        best = None
        best_rate = 0.0
        for c in df.columns:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() < 0.50:
                continue
            rate = num.dropna().between(2000, 2035).mean() if num.notna().any() else 0.0
            if rate > best_rate and rate >= 0.70:
                best_rate = rate
                best = c
        if best is None:
            for c in df.columns:
                if "year" in norm(c) or "time" in norm(c):
                    return c
        return best

    def detect_score_col(df: pd.DataFrame, exclude: set) -> str | None:
        # ưu tiên cột có tên gợi ý
        for c in df.columns:
            if c in exclude:
                continue
            cl = norm(c)
            if ("mci" in cl and "score" in cl) or cl in ("mci", "score", "index", "index score", "overall", "overall score"):
                return c

        best = None
        best_rate = 0.0
        for c in df.columns:
            if c in exclude:
                continue
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() < 0.50:
                continue
            rate = num.dropna().between(-5, 105).mean() if num.notna().any() else 0.0
            if rate > best_rate and rate >= 0.60:
                best_rate = rate
                best = c
        return best

    def detect_name_col(df: pd.DataFrame, exclude: set) -> str | None:
        # ưu tiên cột tên rõ ràng
        for c in df.columns:
            if c in exclude:
                continue
            cl = norm(c)
            if cl in ("country", "economy", "name"):
                return c

        best = None
        best_len = 0.0
        for c in df.columns:
            if c in exclude:
                continue
            s = df[c].dropna().astype(str).str.strip()
            if len(s) < 20:
                continue
            as_num = pd.to_numeric(s, errors="coerce")
            if as_num.notna().mean() > 0.50:
                continue
            avg_len = s.map(len).mean()
            if avg_len > best_len and avg_len >= 6:
                best_len = avg_len
                best = c
        return best

    candidates = []

    for sh in xl.sheet_names:
        try:
            df = auto_header_parse(sh)
        except Exception:
            continue

        if df is None or df.shape[0] < 10 or df.shape[1] < 3:
            continue

        # case WIDE: có nhiều cột năm
        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]
        iso_col = detect_iso_col(df)
        name_col = detect_name_col(df, exclude=set([iso_col]) if iso_col else set())

        # WIDE ưu tiên: có year_cols và có ít nhất name hoặc iso
        if len(year_cols) >= 3 and (iso_col is not None or name_col is not None):
            keep = []
            if iso_col is not None:
                keep.append(iso_col)
            if name_col is not None:
                keep.append(name_col)
            keep += year_cols

            tmp = df[keep].copy()
            tmp = tmp.melt(
                id_vars=[c for c in [iso_col, name_col] if c is not None],
                value_vars=year_cols,
                var_name="year",
                value_name="mci_score",
            )

            if iso_col is not None:
                tmp = tmp.rename(columns={iso_col: "country_iso3"})
            if name_col is not None:
                tmp = tmp.rename(columns={name_col: "country_name_mci"})

            tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
            tmp["mci_score"] = pd.to_numeric(tmp["mci_score"], errors="coerce")
            tmp = tmp.dropna(subset=["year", "mci_score"])

            if "country_iso3" in tmp.columns:
                tmp["country_iso3"] = tmp["country_iso3"].astype(str).str.strip().str.upper()
                tmp.loc[~tmp["country_iso3"].str.fullmatch(r"[A-Z]{3}", na=False), "country_iso3"] = np.nan

            # scale nếu 0-1
            if tmp["mci_score"].max() <= 1.5:
                tmp["mci_score"] = tmp["mci_score"] * 100

            score = 50 + len(year_cols)
            candidates.append((score, sh, tmp))
            continue

        # case LONG
        year_col = detect_year_col(df)
        exclude = set([c for c in [iso_col, year_col] if c is not None])
        score_col = detect_score_col(df, exclude=exclude)
        name_col = detect_name_col(df, exclude=exclude | ({score_col} if score_col else set()))

        if year_col is not None and score_col is not None and (iso_col is not None or name_col is not None):
            keep = []
            if iso_col is not None:
                keep.append(iso_col)
            if name_col is not None:
                keep.append(name_col)
            keep += [year_col, score_col]

            tmp = df[keep].copy()
            if iso_col is not None:
                tmp = tmp.rename(columns={iso_col: "country_iso3"})
            if name_col is not None:
                tmp = tmp.rename(columns={name_col: "country_name_mci"})
            tmp = tmp.rename(columns={year_col: "year", score_col: "mci_score"})

            tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
            tmp["mci_score"] = pd.to_numeric(tmp["mci_score"], errors="coerce")
            tmp = tmp.dropna(subset=["year", "mci_score"])

            if "country_iso3" in tmp.columns:
                tmp["country_iso3"] = tmp["country_iso3"].astype(str).str.strip().str.upper()
                tmp.loc[~tmp["country_iso3"].str.fullmatch(r"[A-Z]{3}", na=False), "country_iso3"] = np.nan

            if tmp["mci_score"].max() <= 1.5:
                tmp["mci_score"] = tmp["mci_score"] * 100

            score = 40
            candidates.append((score, sh, tmp))
            continue

    if not candidates:
        raise ValueError(
            "Không nhận diện được sheet MCI phù hợp. "
            "Gợi ý: MCI file có thể đổi cấu trúc; hãy thử tải thủ công và kiểm tra sheet/cột."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_sheet, out = candidates[0]

    # chuẩn hoá year int
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)

    return out

# -----------------------
# Main compare
# -----------------------
def run_cross_section(dii_panel: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) DII năm 2022
    # -------------------------
    dii_2022 = dii_country_year(dii_panel, 2022)
    dii_2022["rank_dii_2022"] = dense_rank_desc(dii_2022["dii_0_100"])

    # base luôn được khởi tạo từ đây
    base = dii_2022.copy()

    # -------------------------
    # 2) EGDI 2022
    # -------------------------
    egdi = load_egdi(URL_EGDI)
    egdi_2022 = egdi[egdi["year"] == 2022].copy()
    egdi_2022["egdi_score"] = pd.to_numeric(egdi_2022["egdi_score"], errors="coerce")
    egdi_2022["rank_egdi_2022"] = dense_rank_desc(egdi_2022["egdi_score"])

    base = base.merge(
        egdi_2022[["country_iso3", "egdi_score", "rank_egdi_2022"]],
        on="country_iso3",
        how="left",
    )

    # -------------------------
    # 3) NRI 2022 (ISO3 ưu tiên, fallback theo tên)
    # -------------------------
    nri_2022 = load_nri_2022(URL_NRI_2022).copy()
    nri_2022["nri_score"] = pd.to_numeric(nri_2022["nri_score"], errors="coerce")

    # nếu không có rank thì tự tạo rank từ score
    if "nri_rank" not in nri_2022.columns:
        nri_2022["rank_nri_2022"] = dense_rank_desc(nri_2022["nri_score"])
    else:
        nri_2022["rank_nri_2022"] = pd.to_numeric(nri_2022["nri_rank"], errors="coerce")

    # merge theo ISO3 nếu đủ tốt
    has_iso = ("country_iso3" in nri_2022.columns) and nri_2022["country_iso3"].notna().any()

    if has_iso:
        base = base.merge(
            nri_2022[["country_iso3", "nri_score", "rank_nri_2022"]],
            on="country_iso3",
            how="left",
        )
    else:
        # fallback theo country name
        def _norm_name(x: str) -> str:
            x = str(x).strip().lower()
            x = re.sub(r"[^\w\s]", " ", x)
            x = re.sub(r"\s+", " ", x).strip()
            # chuẩn hoá nhẹ vài cụm từ hay gặp
            x = x.replace("republic of", "").replace("the", "").strip()
            return x

        if "country_name_nri" not in nri_2022.columns:
            raise ValueError("NRI dataset thiếu ISO3 và thiếu cả country_name_nri để fallback merge.")

        tmp_dii = base.copy()
        tmp_dii["key_name"] = tmp_dii["country_name"].map(_norm_name)

        tmp_nri = nri_2022.copy()
        tmp_nri["key_name"] = tmp_nri["country_name_nri"].map(_norm_name)

        tmp_dii = tmp_dii.merge(
            tmp_nri[["key_name", "nri_score", "rank_nri_2022"]],
            on="key_name",
            how="left",
        ).drop(columns=["key_name"])

        base = tmp_dii

    # -------------------------
    # 4) MCI: lấy năm lớn nhất có trong file (tự động)
    # -------------------------
    mci = load_mci(URL_MCI_2022)

    # chọn năm mới nhất có trong mci
    y_mci = int(mci["year"].max())
    mci_y = mci[mci["year"] == y_mci].copy()
    mci_y["mci_score"] = pd.to_numeric(mci_y["mci_score"], errors="coerce")
    mci_y["rank_mci"] = dense_rank_desc(mci_y["mci_score"])

    # merge theo ISO3 nếu có
    has_iso_mci = ("country_iso3" in mci_y.columns) and mci_y["country_iso3"].notna().any()

    if has_iso_mci:
        base = base.merge(
            mci_y[["country_iso3", "mci_score", "rank_mci"]],
            on="country_iso3",
            how="left",
        )
    else:
        # fallback theo country name (giống NRI)
        def _norm_name(x: str) -> str:
            x = str(x).strip().lower()
            x = re.sub(r"[^\w\s]", " ", x)
            x = re.sub(r"\s+", " ", x).strip()
            x = x.replace("republic of", "").replace("the", "").strip()
            return x

        if "country_name_mci" not in mci_y.columns:
            raise ValueError("MCI dataset thiếu ISO3 và thiếu cả country_name_mci để fallback merge.")

        tmp_dii = base.copy()
        tmp_dii["key_name"] = tmp_dii["country_name"].map(_norm_name)

        tmp_mci = mci_y.copy()
        tmp_mci["key_name"] = tmp_mci["country_name_mci"].map(_norm_name)

        tmp_dii = tmp_dii.merge(
            tmp_mci[["key_name", "mci_score", "rank_mci"]],
            on="key_name",
            how="left",
        ).drop(columns=["key_name"])

        base = tmp_dii

    # đổi tên cột rank theo năm cho đẹp trong output
    base = base.rename(columns={"rank_mci": f"rank_mci_{y_mci}"})
    base = base.rename(columns={"mci_score": f"mci_score_{y_mci}"})

    # -------------------------
    # 5) Xuất file merge
    # -------------------------
    base.to_csv(outdir / "benchmark_merged_cross_section.csv", index=False)

    # -------------------------
    # 6) Corr summary (Spearman rank là trọng tâm)
    # -------------------------
    rows = []

    rows.append({
        "benchmark": "EGDI_2022",
        "n": int(base[["dii_0_100", "egdi_score"]].dropna().shape[0]),
        "spearman_rank": spearman(base["rank_dii_2022"], base["rank_egdi_2022"]),
        "pearson_score": pearson(base["dii_0_100"], base["egdi_score"]),
    })

    rows.append({
        "benchmark": "NRI_2022",
        "n": int(base[["dii_0_100", "nri_score"]].dropna().shape[0]),
        "spearman_rank": spearman(base["rank_dii_2022"], base["rank_nri_2022"]),
        "pearson_score": pearson(base["dii_0_100"], base["nri_score"]),
    })

    rows.append({
        "benchmark": f"MCI_{y_mci}",
        "n": int(base[["dii_0_100", f"mci_score_{y_mci}"]].dropna().shape[0]),
        "spearman_rank": spearman(base["rank_dii_2022"], base[f"rank_mci_{y_mci}"]),
        "pearson_score": pearson(base["dii_0_100"], base[f"mci_score_{y_mci}"]),
    })

    corr = pd.DataFrame(rows)
    corr.to_csv(outdir / "benchmark_correlations_cross_section.csv", index=False)

    # -------------------------
    # 7) Outliers theo gap rank (top 20 abs gap)
    # -------------------------
    def top_gaps(rank_a, rank_b, label):
        tmp = base[["country_iso3", "country_name", rank_a, rank_b]].dropna().copy()
        tmp[f"gap_{label}"] = tmp[rank_b] - tmp[rank_a]
        tmp["abs_gap"] = tmp[f"gap_{label}"].abs()
        return tmp.sort_values("abs_gap", ascending=False).head(20)

    top_gaps("rank_dii_2022", "rank_egdi_2022", "egdi").to_csv(
        outdir / "outliers_rank_gap_dii_vs_egdi.csv", index=False
    )
    top_gaps("rank_dii_2022", "rank_nri_2022", "nri").to_csv(
        outdir / "outliers_rank_gap_dii_vs_nri.csv", index=False
    )
    top_gaps("rank_dii_2022", f"rank_mci_{y_mci}", f"mci_{y_mci}").to_csv(
        outdir / f"outliers_rank_gap_dii_vs_mci_{y_mci}.csv", index=False
    )

def run_panel_light(dii_panel: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    egdi = load_egdi(URL_EGDI)
    mci = load_mci(URL_MCI_2022)

    # EGDI thường có các năm 2016/2018/2020/2022; MCI có annual (đến 2021)
    years_egdi = sorted(set(egdi["year"].dropna().astype(int).unique()))
    years_mci = sorted(set(mci["year"].dropna().astype(int).unique()))

    # Corr theo từng năm (DII vs EGDI)
    rows = []
    for y in years_egdi:
        d = dii_country_year(dii_panel, int(y))
        e = egdi[egdi["year"] == int(y)][["country_iso3", "egdi_score"]].copy()
        tmp = d.merge(e, on="country_iso3", how="inner")
        rows.append({
            "benchmark": "EGDI",
            "year": int(y),
            "n": int(tmp.dropna().shape[0]),
            "spearman_score": spearman(tmp["dii_0_100"], tmp["egdi_score"]),
            "pearson_score": pearson(tmp["dii_0_100"], tmp["egdi_score"]),
        })

    # Corr theo từng năm (DII vs MCI)
    # chỉ lấy những năm có trong DII panel để tránh rỗng
    years_dii = sorted(set(dii_panel["year"].unique()))
    years_mci = sorted(set(mci["year"].dropna().astype(int).unique()))
    years_overlap = [y for y in years_mci if y in years_dii]

    has_iso_mci_all = ("country_iso3" in mci.columns) and mci["country_iso3"].notna().any()

    for y in years_overlap:
        d = dii_country_year(dii_panel, int(y))

        e = mci[mci["year"] == int(y)].copy()
        if has_iso_mci_all:
            e = e[["country_iso3", "mci_score"]].copy()
            tmp = d.merge(e, on="country_iso3", how="inner")
        else:
            if "country_name_mci" not in e.columns:
                continue

            def _norm_name(x: str) -> str:
                x = str(x).strip().lower()
                x = re.sub(r"[^\w\s]", " ", x)
                x = re.sub(r"\s+", " ", x).strip()
                x = x.replace("republic of", "").replace("the", "").strip()
                return x

            d2 = d.copy()
            d2["key_name"] = d2["country_name"].map(_norm_name)
            e2 = e[["country_name_mci", "mci_score"]].copy()
            e2["key_name"] = e2["country_name_mci"].map(_norm_name)
            tmp = d2.merge(e2[["key_name", "mci_score"]], on="key_name", how="inner")

        rows.append({
            "benchmark": "MCI",
            "year": int(y),
            "n": int(tmp.dropna().shape[0]),
            "spearman_score": spearman(tmp["dii_0_100"], tmp["mci_score"]),
            "pearson_score": pearson(tmp["dii_0_100"], tmp["mci_score"]),
        })

def main(panel: Path, outdir: Path) -> None:
    dii_panel = load_dii_panel(panel)

    run_cross_section(dii_panel, outdir / "cross_section")
    run_panel_light(dii_panel, outdir / "panel_light")

    print("Done. Outputs:", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # base_dir = .../dii_thesis (tính từ vị trí file src/s10_benchmark_compare.py)
    base_dir = Path(__file__).resolve().parents[1]  # dii_thesis/
    default_panel = base_dir / "data" / "processed" / "dii_core" / "dii_core_panel.csv"
    default_outdir = base_dir / "data" / "processed" / "dii_core" / "benchmark"

    p.add_argument("--panel", type=str, default=str(default_panel))
    p.add_argument("--outdir", type=str, default=str(default_outdir))

    args = p.parse_args()
    main(Path(args.panel), Path(args.outdir))

