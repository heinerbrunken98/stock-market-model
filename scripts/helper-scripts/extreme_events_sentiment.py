from pathlib import Path
import pandas as pd
import numpy as np

# paths
BASE = Path("/Users/heiner/stock-market-model")
PRICES_CSV = BASE / "data/sp500_prices/sp500_prices.csv"                
SENT_DAILY = BASE / "data/finbert/total_per_day.parquet"        
OUT_CSV    = BASE / "out/extreme_events.csv"
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

    
    out = out.reset_index(drop=True)
    out.insert(0, "id", out.index + 1)
    out = out[["id", "day", "pct_move", "direction"]]
    # output for latex tables 
    # out["pct_move"] = (out["pct_move"] * 100).round(2).astype(str) + r"\%"

    
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved → {OUT_CSV}")

if __name__ == "__main__":
    main()
    