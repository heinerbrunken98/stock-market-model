import pandas as pd
from pathlib import Path

TOP15 = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_daily_top15.parquet")      
LABELS = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_topic_labels.parquet")    

top15 = pd.read_parquet(TOP15)        
labels = pd.read_parquet(LABELS)      

# if needed, rename columns for consistency
if "topic_id" in labels.columns:
    labels = labels.rename(columns={"topic_id": "topic"})
if "Name" in labels.columns and "topic_label" not in labels.columns:
    labels = labels.rename(columns={"Name": "topic_label"})

top15_lab = top15.merge(labels[["topic","topic_label"]], on="topic", how="left")
print(top15_lab.head())

# saving the labeled top15 topics
top15_lab.to_parquet("/Users/heiner/stock-market-model/data/bertopic/bertopic_daily_top15_labeled.parquet", index=False)
