import pandas as pd
from pathlib import Path

# ---------- Paths ----------
events_path = "/Users/heiner/stock-market-model/out/extreme_events.csv"           
topics_path = "/Users/heiner/stock-market-model/data/bertopic/total_per_day_top15.parquet" 
out_dir = Path("/Users/heiner/stock-market-model/out/events_topics")                             
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
events = pd.read_csv(events_path, parse_dates=["day"])
topics = pd.read_parquet(topics_path)

# Ensure datetime type
if not pd.api.types.is_datetime64_any_dtype(topics["day"]):
    topics["day"] = pd.to_datetime(topics["day"])

# Normalize dates
events["day"] = events["day"].dt.normalize()
topics["day"] = topics["day"].dt.normalize()

# ---------- Filter to March 2018 ----------
start, end = pd.Timestamp("2018-03-01"), pd.Timestamp("2018-03-30")
events = events[(events["day"] >= start) & (events["day"] <= end)].copy()
topics = topics[(topics["day"] >= start) & (topics["day"] <= end)].copy()

# Keep only Top-5 topics per day (without rank)
topics_top5 = topics[topics["rank"] <= 5].copy()
topics_top5 = topics_top5.rename(columns={"topic_label": "topic:"})

# ---------- 1) Event-matched topic table ----------
events_topics = (
    events[["id","day"]]
    .merge(topics_top5[["day","topic:","count"]], on="day", how="inner")
    .sort_values(["day","id"])
    .reset_index(drop=True)
)
events_topics.to_csv(out_dir / "events_topics_top5_mar2018.csv", index=False)

# ---------- Day sets ----------
event_days = set(events["day"].unique())
all_days   = set(topics_top5["day"].unique())
rest_days  = all_days - event_days

topics_event = topics_top5[topics_top5["day"].isin(event_days)].copy()
topics_rest  = topics_top5[topics_top5["day"].isin(rest_days)].copy()

n_event_days = len(event_days)
n_rest_days  = len(rest_days)

# ---------- 2) Frequency tables (Top-5 only, no rank) ----------
def topic_frequency(df, n_days):
    g = df.groupby("topic:", as_index=False).agg(
        freq_days=("day", "nunique"),
        total_count=("count", "sum"),
        avg_count_per_day=("count", "mean"),
    )
    g["share_days"] = g["freq_days"] / max(n_days, 1)
    g = g.sort_values(["freq_days","total_count"], ascending=False)
    return g

freq_event = topic_frequency(topics_event, n_event_days)
freq_rest  = topic_frequency(topics_rest,  n_rest_days)

freq_event.to_csv(out_dir / "topics_frequency_event_days_top5_mar2018.csv", index=False)
freq_rest.to_csv(out_dir / "topics_frequency_rest_days_top5_mar2018.csv",  index=False)

# ---------- Print quick heads ----------
print("Saved:", out_dir / "events_topics_top5_mar2018.csv")
print("Saved:", out_dir / "topics_frequency_event_days_top5_mar2018.csv")
print("Saved:", out_dir / "topics_frequency_rest_days_top5_mar2018.csv")
print("\nEvent-matched Top-5 topics (Mar 2018, first rows):")
print(events_topics.head())
print("\nEvent-day frequency (Mar 2018, Top-5 only):")
print(freq_event.head())
print("\nRest-day frequency (Mar 2018, Top-5 only):")
print(freq_rest.head())
