import pandas as pd
PATH = "/Users/heiner/stock-market-model/data/finbert/total_per_articles.parquet"
df = pd.read_parquet(PATH, engine="pyarrow")
col = "sent_score"
s = df[col]
avg = s.mean()
print(f"Average confidence for sentiment labels: {avg:.6f}")
