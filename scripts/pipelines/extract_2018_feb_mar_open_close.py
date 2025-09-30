#!/usr/bin/env python3
# extract_2018_feb_mar_open_close.py
from pathlib import Path
import pandas as pd

IN_CSV  = Path("/Users/heiner/stock-market-model/data/sp500_prices/sp500_prices_raw.csv")
OUT_CSV = Path("/Users/heiner/stock-market-model/data/sp500_prices/sp500_prices_extracted.csv")

START = "2018-02-01"
END   = "2018-03-30"

# robust read (auto-detect delimiter/quotes)
df = pd.read_csv(IN_CSV, sep=None, engine="python")

# find columns case-insensitively
cols_lut = {c.lower(): c for c in df.columns}
date_col = cols_lut.get("date")
open_col = cols_lut.get("open")
close_col= cols_lut.get("close")
if not all([date_col, open_col, close_col]):
    raise SystemExit(f"Need columns 'date','open','close'. Got: {list(df.columns)}")

# parse date; tolerate mixed formats
df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(), errors="coerce", infer_datetime_format=True)
mask = (df[date_col] >= START) & (df[date_col] <= END)
out = df.loc[mask, [date_col, open_col, close_col]].copy()

# make numeric (handles thousands separators)
for c in [open_col, close_col]:
    out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

# ISO date format & sort
out[date_col] = out[date_col].dt.strftime("%Y-%m-%d")
out = out.sort_values(by=date_col).reset_index(drop=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.rename(columns={date_col: "date", open_col: "open", close_col: "close"}).to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}  rows={len(out)}")
