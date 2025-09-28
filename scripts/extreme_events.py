#!/usr/bin/env python3
# extreme_events.py
# Detect daily close→close moves ≥ 3% and join pre-event (t) news sentiment.

from pathlib import Path
import pandas as pd
import numpy as np

# --- fixed project paths (adapt if you move files) ---
BASE = Path("/Users/heiner/stock-market-model")
PRICES_CSV = BASE / "data/sp500_prices/sp500_prices.csv"                # columns: day, close
SENT_DAILY = BASE / "data/finbert/total_per_day.parquet"        # columns: day, mean_signed, pos_share, neg_share, n_articles
OUT_CSV    = BASE / "out/extreme_events.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

THRESH = 0.02      # 3% absolute move
MAX_EVENTS = 10    # keep top-10 by |pct_move|

def main():
    # Prices → compute same-day open→close move
    px = pd.read_csv(PRICES_CSV, usecols=["date", "open", "close"]).sort_values("date")
    px["day"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    px["pct_move"] = (px["close"] - px["open"]) / px["open"]     # open→close return
    px["open"] = px["open"].round(2)
    px["close"] = px["close"].round(2)
    # print(f"difference:\n {px['pct_move']}")
    px = px[["day", "open", "close", "pct_move"]]

    # Event days: |open→close| ≥ THRESH
    ev = px.loc[px["pct_move"].abs() >= THRESH, ["day", "pct_move", "open", "close"]].copy()
    if len(ev) > MAX_EVENTS:
        ev = ev.reindex(ev["pct_move"].abs().sort_values(ascending=False).head(MAX_EVENTS).index)
    ev = ev.sort_values("day").reset_index(drop=True)
    ev["direction"] = np.where(ev["pct_move"] >= 0, "up", "down")

    # Same-day FinBERT sentiment
    daily = pd.read_parquet(SENT_DAILY)
    daily["day"] = pd.to_datetime(daily["day"]).dt.tz_localize(None)

    # Join same-day sentiment
    out = (ev.merge(daily[["day", "mean_signed", "pos_share", "neg_share", "n_articles"]],
                    on="day", how="left")
             .sort_values("day")
             .reset_index(drop=True))

    out = out[["day", "pct_move", "direction", "open", "close", "mean_signed", "pos_share", "neg_share", "n_articles"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved → {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
    