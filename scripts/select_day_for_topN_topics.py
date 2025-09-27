from pathlib import Path
import pandas as pd

# Paths to your parquet files
P_TOP15   = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_daily_top15_labeled.parquet")
P_LABELS  = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_topic_labels.parquet")

# 1) Load parquet files
top15 = pd.read_parquet(P_TOP15)      # daily top 15 topics (day, topic, count, rank, etc.)
labels = pd.read_parquet(P_LABELS)    # topic id → topic label mapping

# 2) Normalize column names (depends on how parquet was generated)
if "topic_id" in labels.columns:
    labels = labels.rename(columns={"topic_id": "topic"})
if "Name" in labels.columns and "topic_label" not in labels.columns:
    labels = labels.rename(columns={"Name": "topic_label"})

# If top15 does not contain labels, merge them in
if "topic_label" not in top15.columns:
    top15 = top15.merge(labels[["topic", "topic_label"]], on="topic", how="left")

# 3) Convert day to datetime for easier filtering
top15["day"] = pd.to_datetime(top15["day"], errors="coerce")

# 4) Make sure we only keep top 15 per day
#    If no "rank" column → reconstruct top-15 by counts
if "rank" not in top15.columns:
    top15 = (top15.sort_values(["day", "count"], ascending=[True, False])
                   .groupby("day").head(15)
                   .reset_index(drop=True))
else:
    top15 = top15[top15["rank"] <= 15].copy()


# ---------- Helper functions ----------

def top15_for_day(df: pd.DataFrame, day: str | pd.Timestamp) -> pd.DataFrame:
    """
    Get the top-15 topics for a specific day.
    """
    day = pd.to_datetime(day)
    out = (df[df["day"] == day]
             .sort_values(["rank", "count"], ascending=[True, False])
             .loc[:, [c for c in ["day","topic","topic_label","count","share","rank"] if c in df.columns]])
    return out.reset_index(drop=True)

def overall_top_topics(df: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """
    Get the global top-k topics over the full time span, based on counts.
    """
    agg = (df.groupby(["topic","topic_label"], dropna=False)["count"]
             .sum()
             .reset_index()
             .sort_values("count", ascending=False)
             .head(k))
    return agg.reset_index(drop=True)


# ---------- Usage examples ----------

# A) Print available days
days = sorted(top15["day"].dropna().unique())
print("Number of days:", len(days))
print("First day:", days[0], "Last day:", days[-1])

# Example: show top-15 topics for the first available day
d0 = days[15]
print("\nTop-15 topics on", d0.date())
print(top15_for_day(top15, d0))

# B) Global top-15 topics across the entire dataset
print("\nGlobal Top-15 topics (sum of counts):")
print(overall_top_topics(top15, k=15))
