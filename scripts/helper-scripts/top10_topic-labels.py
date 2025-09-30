import pandas as pd

# Datei laden
df = pd.read_parquet("/Users/heiner/stock-market-model/data/bertopic/total_per_day_top15.parquet")

# Überblick verschaffen
print(df.head())

# Häufigkeit jedes Topics zählen
topic_counts = df["topic"].value_counts().reset_index()
topic_counts.columns = ["topic", "Count"]

# Labels dazu mappen
topic_labels = df[["topic", "topic_label"]].drop_duplicates()

# Joinen
top_topics = topic_counts.merge(topic_labels, on="topic", how="left")

# Top 10 auswählen
top10 = top_topics.head(10)

print("Top 10 Topics insgesamt:")
print(top10)

# Nur topic und topic_label speichern (ohne Count)
top10.drop(columns=["Count"]).to_csv(
    "/Users/heiner/stock-market-model/out/top10_topics_overall.csv", 
    index=False
)
