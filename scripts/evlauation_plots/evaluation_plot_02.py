import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Load data
daily  = pd.read_csv("/Users/heiner/stock-market-model/out/daily_joined.csv", parse_dates=["day"])
events = pd.read_csv("/Users/heiner/stock-market-model/out/extreme_events.csv", parse_dates=["day"])

# Merge same day
df = events.merge(daily[["day", "mean_signed"]], on="day", how="left")
top5 = df.reindex(df["pct_move"].abs().sort_values(ascending=False).index).head(5).copy()

# X axis
x = np.arange(len(top5))
w = 0.25   # narrower bars
labels = [f"{d.date()}" for d in top5["day"]]
c1, c2 = "tab:blue", "tab:red"

# Use constrained_layout for tighter layout
fig, ax1 = plt.subplots(figsize=(8,4.5), constrained_layout=True)
ax2 = ax1.twinx()

# Bars
ax1.bar(x - w/2, top5["mean_signed"].values, w, color=c1, label="Sentiment", bottom=0)
ax2.bar(x + w/2, top5["pct_move"].values,    w, color=c2, label="Move (%)",  bottom=0)

# Symmetric limits
def symmetric_limits(series, pad=0.1):
    amax = float(np.nanmax(np.abs(series)))
    lim  = amax * (1.0 + pad) if amax > 0 else 1.0
    return -lim, lim

ax1.set_ylim(*symmetric_limits(top5["mean_signed"]))
ax2.set_ylim(*symmetric_limits(top5["pct_move"]))

# Labels
ax1.set_ylabel("Sentiment (score)", color=c1)
ax2.set_ylabel("Move (%)", color=c2)
ax1.tick_params(axis="y", labelcolor=c1)
ax2.tick_params(axis="y", labelcolor=c2)

# X ticks
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right")

# Grid + zero line
ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax1.axhline(0, color="black", linewidth=1)

# plt.title("Sentiment vs. Price Move on Extreme Events (Top 5)")

handles, labels = [], []
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    handles += h; labels += l
ax1.legend(handles, labels, loc="upper right")

# Even tighter control (optional fine-tune)
plt.tight_layout(pad=0.5)

out_path = "/Users/heiner/stock-market-model/out/events_sentiment_vs_pctmove_dualaxis.png"
fig.savefig(out_path, dpi=150)
plt.show()
