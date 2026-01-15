from pathlib import Path

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"
SRC_DIR = BASE_DIR / "src"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# World Bank API settings
# -----------------------------
WDI_BASE = "https://api.worldbank.org/v2"

# Giai đoạn nghiên cứu chính (luận văn)
STUDY_START_YEAR = 2015
STUDY_END_YEAR = 2022

# Khoảng năm dùng để fetch từ WB API (inclusive). Giữ đúng giai đoạn chính để pipeline tái lập.
DATE_RANGE = f"{STUDY_START_YEAR}:{STUDY_END_YEAR}"

# Cửa sổ dùng để báo cáo chất lượng dữ liệu (DQ)
DQ_WINDOW = (STUDY_START_YEAR, STUDY_END_YEAR)

# Lọc aggregates: các entity WB không phải quốc gia thường có region.value == "Aggregates"
EXCLUDE_REGION_VALUE = "Aggregates"

# -----------------------------
# DII-Core (WDI-only) definition
# -----------------------------
# 6 biến lõi để xây DII-Core (bám logic: access/adoption - infra/capacity - human capital readiness)
DII_CORE_INDICATORS = [
    "IT.NET.USER.ZS",   # Individuals using the Internet (% of population)
    "IT.CEL.SETS.P2",   # Mobile cellular subscriptions (per 100 people)
    "IT.NET.BBND.P2",   # Fixed broadband subscriptions (per 100 people)
    "IT.NET.SECR.P6",   # Secure Internet servers (per 1 million people) -> log1p
    "SE.SEC.ENRR",      # School enrollment, secondary (gross, %)
    "SE.TER.ENRR",      # School enrollment, tertiary (gross, %)
]

DII_PILLARS = {
    "pillar_access_adoption": ["IT.NET.USER.ZS", "IT.CEL.SETS.P2"],
    "pillar_infra_capacity":  ["IT.NET.BBND.P2", "IT.NET.SECR.P6"],
    "pillar_human_capital":   ["SE.SEC.ENRR", "SE.TER.ENRR"],
}

# Missingness rule (luận văn - minh bạch, khả thi)
# - Trụ được tính nếu có >= 1/2 chỉ báo trong trụ
# - DII được tính nếu có >= 2/3 trụ
MIN_INDICATORS_PER_PILLAR = 1
MIN_PILLARS_FOR_DII = 2
