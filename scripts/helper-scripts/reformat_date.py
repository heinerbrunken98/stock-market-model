from pathlib import Path
import pandas as pd

# === BERTopic ===
# 02_articles_top15_labeled.parquet -> 02_articles_top15_labeled_clean.parquet
# inp = Path("/Users/heiner/stock-market-model/data/bertopic/02_articles_top15_labeled.parquet")
# out = Path("/Users/heiner/stock-market-model/data/bertopic/02_articles_top15_labeled_clean.parquet")

# 03_articles_top15_labeled.parquet -> 02_articles_top15_labeled_clean.parquet
# inp = Path("/Users/heiner/stock-market-model/data/bertopic/03_articles_top15_labeled.parquet")
# out = Path("/Users/heiner/stock-market-model/data/bertopic/03_articles_top15_labeled_clean.parquet")

# === FinBERT ===
# 02_per_articles.parquet -> 02_per_articles_clean.parquet
# inp = Path("/Users/heiner/stock-market-model/data/finbert/02_per_articles.parquet")
# out = Path("/Users/heiner/stock-market-model/data/finbert/02_per_articles_clean.parquet")

# 02_per_day_finbert.parquet -> 02_per_day_finbert_clean.parquet
# inp = Path("/Users/heiner/stock-market-model/data/finbert/02_per_day_finbert.parquet")
# out = Path("/Users/heiner/stock-market-model/data/finbert/02_per_day_finbert_clean.parquet")

# 03_per_articles.parquet -> 03_per_articles_clean.parquet
# inp = Path("/Users/heiner/stock-market-model/data/finbert/03_per_articles.parquet")
# out = Path("/Users/heiner/stock-market-model/data/finbert/03_per_articles_clean.parquet")

# 03_per_day_finbert.parquet -> 03_per_day_finbert_clean.parquet
# inp = Path("/Users/heiner/stock-market-model/data/finbert/03_per_day_finbert.parquet")
# out = Path("/Users/heiner/stock-market-model/data/finbert/03_per_day_finbert_clean.parquet")

inp = Path("/Users/heiner/stock-market-model/data/bertopic/new/total_daily_top15.parquet")
out = Path("/Users/heiner/stock-market-model/data/bertopic/new/total_daily_top15_clean.parquet")

# -----------------------

df = pd.read_parquet(inp)

# bring day to YYYY-MM-DD format
if pd.api.types.is_integer_dtype(df["day"]):
    df["day"] = pd.to_datetime(df["day"], unit="ms", utc=True)
df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
# create string date 
df["day"] = df["day"].dt.strftime("%Y-%m-%d")

# df = df.rename(columns={"Count": "topic_global_count"})  # sprechender Name
# df = df.sort_values(["day", "rank"])

out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print("✅ saved:", out)
