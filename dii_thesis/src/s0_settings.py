from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"
SRC_DIR = BASE_DIR / "src"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

WDI_BASE = "https://api.worldbank.org/v2"
DATE_RANGE = "2010:2023"  # bạn có thể chỉnh
DQ_WINDOW = (2018, 2022)

# Lọc aggregates: các entity WB không phải quốc gia thường có region.value == "Aggregates"
EXCLUDE_REGION_VALUE = "Aggregates"
