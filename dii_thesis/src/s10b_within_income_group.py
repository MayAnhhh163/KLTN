from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{where}] Thiếu cột bắt buộc: {miss}")


def _bench_cols(df: pd.DataFrame) -> dict[str, str]:
    out = {}
    if "egdi_score" in df.columns:
        out["EGDI"] = "egdi_score"
    if "nri_score" in df.columns:
        out["NRI"] = "nri_score"
    for c in df.columns:
        if str(c).startswith("mci_score_"):
            out[c.replace("mci_score_", "MCI_")] = c
    if not out:
        raise ValueError("Không thấy cột benchmark score (egdi_score / nri_score / mci_score_YYYY) trong file merge.")
    return out


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(tmp)
    if n < 10:
        return float("nan"), float("nan"), n
    rho, p = spearmanr(tmp["x"], tmp["y"])
    return float(rho), float(p), n


def main(merged_csv: Path, outdir: Path, year: int = 2022) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)
    _require_cols(df, ["year", "dii_0_100", "income_group"], "merged")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"] == year].copy()

    bench = _bench_cols(df)

    # baseline
    df["dii_baseline"] = pd.to_numeric(df["dii_0_100"], errors="coerce")

    # strict: chỉ giữ quan sát đủ 3 trụ (nếu có cột)
    have_strict = "n_pillars_available" in df.columns
    if have_strict:
        n_p = pd.to_numeric(df["n_pillars_available"], errors="coerce")
        df["dii_strict"] = df["dii_baseline"].where(n_p >= 3, np.nan)

    rows = []

    for bname, bcol in bench.items():
        for g, sub in df.groupby("income_group"):
            rho, p, n = _spearman(sub["dii_baseline"], sub[bcol])
            rows.append({
                "dii_variant": "baseline",
                "income_group": g,
                "benchmark": bname,
                "n": n,
                "spearman_rho": rho,
                "p_value": p,
            })

            if have_strict:
                rho2, p2, n2 = _spearman(sub["dii_strict"], sub[bcol])
                rows.append({
                    "dii_variant": "strict",
                    "income_group": g,
                    "benchmark": bname,
                    "n": n2,
                    "spearman_rho": rho2,
                    "p_value": p2,
                })

    table_4x = pd.DataFrame(rows).sort_values(["dii_variant", "benchmark", "income_group"])
    out_path = outdir / "table_4x_within_income_group_spearman.csv"
    table_4x.to_csv(out_path, index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--merged_csv",
        type=str,
        default=str(Path("dii_thesis") / "data" / "processed" / "dii_core" / "benchmark" / "cross_section" / "benchmark_merged_cross_section.csv"),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path("dii_thesis") / "data" / "processed" / "dii_core" / "benchmark" / "cross_section"),
    )
    p.add_argument("--year", type=int, default=2022)
    args = p.parse_args()
    main(Path(args.merged_csv), Path(args.outdir), year=args.year)
