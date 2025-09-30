from pathlib import Path
import pandas as pd
import numpy as np

# paths
BASE = Path("/Users/heiner/stock-market-model")
PRICES_CSV = BASE / "data/sp500_prices/sp500_prices.csv"                
SENT_DAILY = BASE / "data/finbert/total_per_day.parquet"        
OUT_CSV    = BASE / "out/extreme_events_s_t-1.xlsx"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# N% absolute move
THRESH = 0.02 
MAX_EVENTS = 10 

def main():
    # --- prices (day t) ---
    px = pd.read_csv(PRICES_CSV, usecols=["date", "open", "close"]).sort_values("date")
    px["day"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    px["pct_move"] = (px["close"] - px["open"]) / df["open"] if False else (px["close"] - px["open"]) / px["open"]
    px["open"] = px["open"].round(2)
    px["close"] = px["close"].round(2)
    px = px[["day", "open", "close", "pct_move"]].sort_values("day").reset_index(drop=True)

    # --- extreme event days (by day t move) ---
    ev = px.loc[px["pct_move"].abs() >= THRESH, ["day", "pct_move", "open", "close"]].copy()
    if len(ev) > MAX_EVENTS:
        ev = ev.reindex(ev["pct_move"].abs().sort_values(ascending=False).head(MAX_EVENTS).index)
    ev = ev.sort_values("day").reset_index(drop=True)
    ev["direction"] = np.where(ev["pct_move"] >= 0, "up", "down")

    # --- sentiment (use last available day strictly BEFORE t) ---
    daily = pd.read_parquet(SENT_DAILY)
    daily["day"] = pd.to_datetime(daily["day"]).dt.tz_localize(None)
    daily = daily.sort_values("day").rename(columns={"day": "day_sent"})

    # as-of join: for each price/event day t, take sentiment from the latest day < t
    out = pd.merge_asof(
        ev.sort_values("day"),
        daily[["day_sent", "mean_signed", "pos_share", "neg_share", "n_articles"]].sort_values("day_sent"),
        left_on="day",
        right_on="day_sent",
        direction="backward",
        allow_exact_matches=False,   # strictly t-1 (no same-day)
    )

    # optional: rename to make t-1 explicit
    out = out.rename(columns={
        "mean_signed": "mean_signed_tminus1",
        "pos_share": "pos_share_tminus1",
        "neg_share": "neg_share_tminus1",
        "n_articles": "n_articles_tminus1",
    })

    out = out[[
        "day", "pct_move", "direction", "open", "close",
        "mean_signed_tminus1", "pos_share_tminus1", "neg_share_tminus1", "n_articles_tminus1",
        "day_sent"  # keep to see which prior day was used
    ]]

    OUT_XLSX = OUT_CSV.with_suffix(".xlsx")
    out.to_excel(OUT_XLSX, index=False)
    print(f"✅ Saved → {OUT_XLSX}")

if __name__ == "__main__":
    main()
    