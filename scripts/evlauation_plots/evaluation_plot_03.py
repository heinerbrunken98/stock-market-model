import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Output folder
OUT_DIR = Path("/Users/heiner/stock-market-model/out/Figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files 
ev_feb_path = "/Users/heiner/stock-market-model/out/events_topics/topics_frequency_event_days_top5_feb2018.csv"
ev_mar_path = "/Users/heiner/stock-market-model/out/events_topics/topics_frequency_event_days_top5_mar2018.csv"
re_feb_path = "/Users/heiner/stock-market-model/out/events_topics/topics_frequency_rest_days_top5_feb2018.csv"
re_mar_path = "/Users/heiner/stock-market-model/out/events_topics/topics_frequency_rest_days_top5_mar2018.csv"

TOP_K = 12
COLOR_EVENT = "#ff91a4"  
COLOR_REST  = "#1f77b4"  

# loadev_feb = pd.read_csv(ev_feb_path)
ev_mar = pd.read_csv(ev_mar_path)
re_feb = pd.read_csv(re_feb_path)
re_mar = pd.read_csv(re_mar_path)

# normalize column names
def normalize_topic_col(df):
    if "topic:" in df.columns:
        df = df.rename(columns={"topic:": "topic"})
    elif "topic_label" in df.columns:
        df = df.rename(columns={"topic_label": "topic"})
    for c in ("freq_days", "total_count", "avg_count_per_day", "share_days"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

ev_feb = normalize_topic_col(ev_feb)
ev_mar = normalize_topic_col(ev_mar)
re_feb = normalize_topic_col(re_feb)
re_mar = normalize_topic_col(re_mar)

# topic cleaner 
STOP = set("""
the and of in for on to by with at from a an or as about into over under between during including
until against among through above below out up down off again further then once
der die das und von zu mit im am den des als für auf bei aus nach vor ist sind war waren wird werden
de la le les et du des en au aux
""".split())

def simplify_topic(label: str, max_words: int = 3) -> str:
    if not isinstance(label, str):
        return ""
    label = re.sub(r"^\d+[_\-\.\s]+", " ", label)           # drop numeric prefixes
    txt = re.sub(r"[^A-Za-zÄÖÜäöüß ]+", " ", label).lower() # keep letters/spaces
    words = [w for w in txt.split() if w and w not in STOP]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= max_words:
            break
    return " ".join(out) if out else label.strip()

for df in (ev_feb, ev_mar, re_feb, re_mar):
    df["topic_simple"] = df["topic"].apply(simplify_topic)

# Infer number of days per universe from share_days ≈ freq_days / N 
def infer_num_days(df: pd.DataFrame) -> int:
    if "share_days" not in df.columns or df["share_days"].fillna(0).eq(0).all():
        return 0
    s = df.loc[df["share_days"] > 0, ["freq_days", "share_days"]]
    if s.empty:
        return 0
    vals = s["freq_days"] / s["share_days"]
    vals = vals[np.isfinite(vals)]
    return int(round(np.median(vals))) if len(vals) else 0

n_ev_days_total   = infer_num_days(ev_feb) + infer_num_days(ev_mar)
n_rest_days_total = infer_num_days(re_feb) + infer_num_days(re_mar)

# Aggregate over both months per universe 
def aggregate_two_months(df_a: pd.DataFrame, df_b: pd.DataFrame, prefix: str) -> pd.DataFrame:
    both = pd.concat([df_a, df_b], ignore_index=True)
    g = both.groupby("topic_simple", as_index=False).agg(
        days=("freq_days", "sum"),
        total=("total_count", "sum")
    )
    return g.rename(columns={"days": f"{prefix}_days", "total": f"{prefix}_total_count"})

ev_all = aggregate_two_months(ev_feb, ev_mar, "event")
re_all = aggregate_two_months(re_feb, re_mar, "rest")

# Intersection (topics present in both worlds) 
overlap = ev_all.merge(re_all, on="topic_simple", how="inner")

# Normalize to average counts per day (fair comparison) 
overlap["avg_count_per_event_day"] = overlap["event_total_count"] / max(n_ev_days_total, 1)
overlap["avg_count_per_rest_day"]  = overlap["rest_total_count"]  / max(n_rest_days_total, 1)

# Ranking metric: total presence across both universes 
overlap["total_presence_days"] = overlap["event_days"] + overlap["rest_days"]

# Sort & save summary CSV 
overlap_sorted = overlap.sort_values(
    ["total_presence_days", "event_total_count", "rest_total_count"],
    ascending=False
).reset_index(drop=True)

summary_cols = [
    "topic_simple",
    "event_days", "rest_days",
    "event_total_count", "rest_total_count",
    "avg_count_per_event_day", "avg_count_per_rest_day",
    "total_presence_days"
]
summary_path = OUT_DIR / "overlap_event_vs_rest_summary.csv"
overlap_sorted[summary_cols].to_csv(summary_path, index=False)

# Plot: grouped bars of average counts per day 
plot_df = overlap_sorted.head(TOP_K).copy()
x = np.arange(len(plot_df)); w = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, plot_df["avg_count_per_event_day"].values, w, label="Event days", color=COLOR_EVENT)
ax.bar(x + w/2, plot_df["avg_count_per_rest_day"].values,  w, label="Rest days",  color=COLOR_REST)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["topic_simple"].tolist(), rotation=30, ha="right")
ax.set_ylabel("Average topic count per day")
# ax.set_title("Overlapping topics: average count per day (Event vs. Rest)")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plot_path = OUT_DIR / "overlap_event_vs_rest_avgcount.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)

print("Saved:")
print("  CSV :", summary_path)
print("  Plot:", plot_path)
