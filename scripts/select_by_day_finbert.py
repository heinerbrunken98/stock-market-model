# scripts/inspect_daily_finbert.py
from pathlib import Path
import pandas as pd
import numpy as np

# Path to your FinBERT daily parquet (from your finbert_daily pipeline)
P_FINBERT = Path("/Users/heiner/stock-market-model/data/finbert/daily_finbert.parquet")
# expected columns: day, n_articles, sent_label, sent_score, sent_score_signed

# 1) Load & normalize
df = pd.read_parquet(P_FINBERT)
# ensure datetime
df["day"] = pd.to_datetime(df["day"], errors="coerce")
# optional sort
df = df.sort_values("day").reset_index(drop=True)

# ---------- Helper functions ----------

def sentiment_for_day(df: pd.DataFrame, day: str | pd.Timestamp) -> pd.DataFrame:
    """
    Return the FinBERT daily sentiment row for a specific day.
    If multiple rows per day exist, it aggregates them (shouldn't happen with your pipeline).
    """
    day = pd.to_datetime(day)
    sub = df[df["day"] == day]
    if sub.empty:
        raise ValueError(f"No rows for day={day.date()}")
    # If there were multiple rows (unlikely), aggregate sensibly:
    if len(sub) > 1:
        # majority label, mean scores weighted by n_articles
        maj_label = (
            sub.groupby("sent_label")["n_articles"]
               .sum()
               .sort_values(ascending=False)
               .index[0]
        )
        w = sub["n_articles"].clip(lower=1)
        out = pd.DataFrame([{
            "day": day.normalize(),
            "n_articles": int(sub["n_articles"].sum()),
            "sent_label": maj_label,
            "sent_score": float(np.average(sub["sent_score"], weights=w)),
            "sent_score_signed": float(np.average(sub["sent_score_signed"], weights=w)),
        }])
        return out
    # Normal case: exactly one row
    return sub.loc[:, ["day","n_articles","sent_label","sent_score","sent_score_signed"]]


def overall_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Global aggregates across the full time span.
    - label distribution
    - average scores (simple and article-weighted)
    """
    # label distribution
    label_counts = df["sent_label"].value_counts().rename_axis("sent_label").reset_index(name="days")

    # simple (unweighted) averages across days
    avg_simple = df[["sent_score","sent_score_signed"]].mean().rename("avg_simple")

    # article-weighted averages across days (days with more news count more)
    w = df["n_articles"].clip(lower=1)
    avg_weighted = pd.Series({
        "sent_score": float(np.average(df["sent_score"], weights=w)),
        "sent_score_signed": float(np.average(df["sent_score_signed"], weights=w)),
    }, name="avg_weighted")

    # combine into a small report object
    report = {
        "label_distribution": label_counts,
        "averages": pd.concat([avg_simple, avg_weighted], axis=1),
        "n_days": int(df["day"].nunique()),
        "total_articles": int(df["n_articles"].sum()),
    }
    return report


def window_sentiment(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Aggregate over a date window [start, end] inclusive.
    Returns a one-row summary similar to overall, but for a sub-period.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    sub = df[(df["day"] >= start) & (df["day"] <= end)]
    if sub.empty:
        raise ValueError(f"No rows in window {start.date()}..{end.date()}")

    w = sub["n_articles"].clip(lower=1)
    maj_label = (
        sub.groupby("sent_label")["n_articles"]
           .sum()
           .sort_values(ascending=False)
           .index[0]
    )
    out = pd.DataFrame([{
        "start": start.normalize(),
        "end": end.normalize(),
        "days": int(sub["day"].nunique()),
        "n_articles": int(sub["n_articles"].sum()),
        "majority_label": maj_label,
        "avg_sent_score": float(np.average(sub["sent_score"], weights=w)),
        "avg_sent_score_signed": float(np.average(sub["sent_score_signed"], weights=w)),
    }])
    return out


# ---------- Usage examples ----------
if __name__ == "__main__":
    # A) One specific day (dN)
    # pick first available day
    # dN = df["day"].min()
    dN = pd.Timestamp("2018-01-19") # choose specific date
    # print("First day:", dN.date())
    print(sentiment_for_day(df, dN))

    # B) Global aggregates
    rep = overall_sentiment(df)
    print("\nLabel distribution (days):")
    print(rep["label_distribution"])
    print("\nAverages (columns: simple, weighted):")
    print(rep["averages"])
    print(f"\nDays: {rep['n_days']} | Total articles: {rep['total_articles']}")

    # C) Window/period example
    # change the dates to your dataset range
    print("\nWindow aggregate:")
    print(window_sentiment(df, "2018-01-01", "2018-05-31"))
