import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH_PRICES = Path("/Users/heiner/stock-market-model/data/sp500_prices/sp500_prices.csv")
CSV_PATH_SENTIMENT = Path("/Users/heiner/stock-market-model/out/daily_joined.csv")
OUT_PNG  = Path("/Users/heiner/stock-market-model/out/sent_vs_ret_eod.png")

# adjust style for readability
plt.rcParams.update({
    "font.size": 12,          # Basisschriftgröße (alles)
    "axes.labelsize": 16,     # Achsentitel
    "axes.titlesize": 16,     # Plot-Titel
    "xtick.labelsize": 14,    # x-Achsen Tick-Labels
    "ytick.labelsize": 14,    # y-Achsen Tick-Labels
    "legend.fontsize": 16     # Legende
})


# load
df_prices = pd.read_csv(CSV_PATH_PRICES)
df_prices["date"] = pd.to_datetime(df_prices["date"], format='mixed').dt.tz_localize(None)

df_sentiment = pd.read_csv(CSV_PATH_SENTIMENT)
df_sentiment["day"] = pd.to_datetime(df_sentiment["day"])

# same-day end-of-day return (open -> close)
df_prices["ret"] = (df_prices["close"] - df_prices["open"]) / df_prices["open"]

# correlations on overlapping days (no robustness)
s = df_sentiment[["mean_signed", "ret"]].dropna()
pearson  = s["mean_signed"].corr(s["ret"])
spearman = s["mean_signed"].rank().corr(s["ret"].rank())

# colors
c1, c2 = "tab:blue", "tab:red"

fig, ax1 = plt.subplots(figsize=(12, 5.5))

# line 1: sentiment (left axis)
l1, = ax1.plot(df_sentiment["day"], df_sentiment["mean_signed"], color=c1, label="mean_signed (t)")
ax1.set_ylabel("Sentiment (day t)", color=c1)
ax1.tick_params(axis="y", colors=c1)
ax1.grid(True, alpha=0.25)

# line 2: same-day EOD return (right axis)
ax2 = ax1.twinx()
l2, = ax2.plot(df_prices["date"], df_prices["ret"], color=c2, label="Return open-close (t)")
ax2.set_ylabel("Return open-close (t)", color=c2)
ax2.tick_params(axis="y", colors=c2)

# symmetric y-lims around zero for both axes
def symmetric_limits(series, pad=1.10):
    m = np.nanmax(np.abs(series.values))
    if not np.isfinite(m) or m == 0:
        m = 1.0
    return (-m*pad, m*pad)

ax1.set_ylim(*symmetric_limits(df_sentiment["mean_signed"]))
ax2.set_ylim(*symmetric_limits(df_prices["ret"]))

# single zero line
ax1.axhline(0, color="0.35", lw=1, alpha=0.6)

# color spines to match
ax1.spines["left"].set_color(c1)
ax2.spines["right"].set_color(c2)

# legend
lines = [l1, l2]
ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right")

plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
# OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
# lt.savefig(OUT_PNG, dpi=150)
plt.close()

print(f"Pearson={pearson:.3f} · Spearman={spearman:.3f}")
print(f"Saved -> {OUT_PNG}")
