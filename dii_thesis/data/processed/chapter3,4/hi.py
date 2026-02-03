import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

IN = Path("benchmark_merged_cross_section.csv")
OUT = Path("table_B1_LOO_top20_MCI.csv")

df = pd.read_csv(IN)

# 1) Nhận diện cột year, country, DII, mci một cách an toàn
def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

year_col = pick_col(["year", "Year"])
country_col = pick_col(["country_name", "Country", "country", "economy", "Economy", "name"])
iso_col = pick_col(["iso3", "ISO3", "iso_code"])

# DII thường là 'dii' hoặc 'DII' hoặc 'dii_core'
dii_col = None
for c in df.columns:
    if c.lower() in ["dii", "dii_core", "dii_score"] or ("dii" in c.lower() and "rank" not in c.lower()):
        dii_col = c
        break

# mci thường có 'mci' + 'score' hoặc chỉ 'mci'
mci_col = None
for c in df.columns:
    cl = c.lower()
    if "mci" in cl and ("score" in cl or cl == "mci"):
        mci_col = c
        break
if mci_col is None:
    # fallback: bất kỳ cột nào có 'mci'
    for c in df.columns:
        if "mci" in c.lower():
            mci_col = c
            break

if year_col is None or country_col is None or dii_col is None or mci_col is None:
    raise ValueError(
        f"Không nhận diện được cột cần thiết. year={year_col}, country={country_col}, dii={dii_col}, mci={mci_col}. "
        f"Columns={df.columns.tolist()}"
    )

# 2) Lọc năm cross-section của mci (thường là 2022)
# Nếu file của bạn đã là cross-section thì bỏ lọc year cũng được, nhưng vẫn để an toàn.
target_year = 2022
if year_col in df.columns:
    sub = df[df[year_col] == target_year].copy()
else:
    sub = df.copy()

sub = sub[[country_col, dii_col, mci_col] + ([iso_col] if iso_col else [])].copy()
sub = sub.dropna(subset=[dii_col, mci_col])

# 3) Tính Spearman baseline
base_rho = spearmanr(sub[dii_col], sub[mci_col]).statistic

# 4) Leave-one-out influence
rows = []
for i in range(len(sub)):
    tmp = sub.drop(sub.index[i])
    rho_i = spearmanr(tmp[dii_col], tmp[mci_col]).statistic
    delta = rho_i - base_rho
    rows.append({
        "country_name": sub.iloc[i][country_col],
        "delta": float(delta),
        "abs_delta": float(abs(delta)),
    })

loo = pd.DataFrame(rows).sort_values("abs_delta", ascending=False).head(20)
loo.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Baseline Spearman:", base_rho)
print(loo.head(10))
