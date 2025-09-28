import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("/Users/heiner/stock-market-model/out/eval/daily_joined.csv")
OUT_PNG  = Path("/Users/heiner/stock-market-model/out/eval/sent_vs_ret_t1.png")

# --- load & basic prep ---
df = pd.read_csv(CSV_PATH)
df["day"] = pd.to_datetime(df["day"])

# find sentiment column (be robust to naming)
sent_col = df.columns("mean_signed")

# ensure next-day return exists; if not, derive from same-day ret
def add_ret_t1_from_prices_strict(df: pd.DataFrame, prices_csv: str | Path) -> pd.DataFrame:
    # load prices (strict column names)
    px = pd.read_csv(prices_csv, usecols=["day", "close"])
    px["day"] = pd.to_datetime(px["day"]).dt.tz_localize(None)
    px = px.sort_values("day").reset_index(drop=True)

    # Build mapping to the next trading day
    # next available close
    px["close_next"] = px["close"].shift(-1)
    # next-day return
    px["ret_t1"] = (px["close_next"] - px["close"]) / px["close"]

    # merge onto your dataframe
    out = df.copy()
    if "day" not in out.columns:
        raise ValueError("Your DataFrame must have a 'day' column.")
    out["day"] = pd.to_datetime(out["day"]).dt.tz_localize(None)

    # with left join: keep only the 'ret_t1' we need
    out = out.merge(px[["day", "ret_t1"]], on="day", how="left")

    return out

# pearson & spearman correlations
s = df[[sent_col, "ret_t1"]].dropna()
pearson  = s[sent_col].corr(s["ret_t1"])
spearman = s[sent_col].rank().corr(s["ret_t1"].rank())

# distinct colors for the two lines
c1, c2 = "tab:blue", "tab:red"   

fig, ax1 = plt.subplots(figsize=(12, 5))

# line 1: sentiment (left axis)
l1, = ax1.plot(df["day"], df[sent_col], color=c1, label=f"{sent_col} (t)")
ax1.set_ylabel("Sentiment (day t)", color=c1)
ax1.tick_params(axis="y", colors=c1)
ax1.grid(True, alpha=0.25)

# line 2: next-day return (right axis)
ax2 = ax1.twinx()
l2, = ax2.plot(df["day"], df["ret_t1"], color=c2, label="Return (t+1)")
ax2.set_ylabel("Return (t+1)", color=c2)
ax2.tick_params(axis="y", colors=c2)

# align zero lines by using symmetric limits around 0 on BOTH axes
def symmetric_limits(series, pad=1.05):
    m = np.nanmax(np.abs(series.values))
    m = m if np.isfinite(m) and m > 0 else 1.0
    return (-m*pad, m*pad)

y1_lo, y1_hi = symmetric_limits(df[sent_col], pad=1.10)
y2_lo, y2_hi = symmetric_limits(df["ret_t1"], pad=1.10)
ax1.set_ylim(y1_lo, y1_hi)
ax2.set_ylim(y2_lo, y2_hi)

# draw zero lines on both axes (they will overlap exactly)
ax1.axhline(0, color="0.35", lw=1, alpha=0.6)
ax2.axhline(0, color="0.35", lw=1, alpha=0.6)

# color the spines to match the data
ax1.spines["left"].set_color(c1)
ax2.spines["right"].set_color(c2)

# combined legend
lines = [l1, l2]
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="upper left")

plt.title(f"Sentiment (t) vs. S&P 500 Return (t+1)\n"
          f"Pearson={pearson:.3f} · Spearman={spearman:.3f}")
plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=150)
plt.close()

print(f"Saved → {OUT_PNG}")
