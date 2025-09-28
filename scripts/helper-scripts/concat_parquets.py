#!/usr/bin/env python3
# concat_two_parquets.py
from pathlib import Path
import pandas as pd

# --- fixed paths ---
INP1 = Path("/Users/heiner/stock-market-model/data/finbert/02_per_articles_clean.parquet")
INP2 = Path("/Users/heiner/stock-market-model/data/finbert/03_per_articles_clean.parquet")
OUTP = Path("/Users/heiner/stock-market-model/data/finbert/total_per_articles_clean.parquet")

# optional: set to True if you want to sort by 'day' after concat
SORT_BY_DAY = True

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    if not INP1.exists() or not INP2.exists():
        raise SystemExit(f"Input missing:\n  {INP1} exists={INP1.exists()}\n  {INP2} exists={INP2.exists()}")

    df1 = pd.read_parquet(INP1)
    df2 = pd.read_parquet(INP2)

    # quick schema sanity check
    if list(df1.columns) != list(df2.columns):
        print("[warn] Columns differ (order or names). Proceeding with union and concat.")
        # This still works with sort=False below.

    df = pd.concat([df1, df2], ignore_index=True, sort=False)

    if SORT_BY_DAY and "day" in df.columns:
        try:
            df = df.sort_values("day").reset_index(drop=True)
        except Exception:
            # if day has mixed types, leave as-is
            print("[warn] Could not sort by 'day' (mixed dtype) — leaving original order.")

    ensure_parent(OUTP)
    df.to_parquet(OUTP, index=False)

    print(f"✅ saved → {OUTP.resolve()}")
    print(f"rows: {len(df)} (inp1={len(df1)}, inp2={len(df2)})")
    print(f"cols: {list(df.columns)}")

if __name__ == "__main__":
    main()
