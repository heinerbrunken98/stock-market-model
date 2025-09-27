import pandas as pd
from pathlib import Path

IN_TOPICS   = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_daily_top15.parquet")       
IN_LABELS   = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_topic_labels.parquet")      
OUT_CLEAN   = Path("/Users/heiner/stock-market-model/data/bertopic/bertopic_daily_top15_labeled.parquet")

def load_labels(path: Path) -> pd.DataFrame:
    lab = pd.read_parquet(path)
    # Spalten normalisieren
    ren = {}
    if "topic_id" in lab.columns: ren["topic_id"] = "topic"
    if "Topic"    in lab.columns: ren["Topic"]    = "topic"
    if "Name"     in lab.columns: ren["Name"]     = "topic_label"
    if ren:
        lab = lab.rename(columns=ren)
    # nur benötigte Spalten
    keep = [c for c in ["topic","topic_label","Count"] if c in lab.columns]
    return lab[keep].drop_duplicates()

def main():
    df = pd.read_parquet(IN_TOPICS)
    labels = load_labels(IN_LABELS)

    # Relevante Spalten auswählen
    base_cols = [c for c in ["day","topic","count","share","rank"] if c in df.columns]
    df = df[base_cols].copy()

    # Sicherstellen: wirklich Top-15 pro Tag
    if "rank" in df.columns:
        df = df[df["rank"] <= 15].copy()
    else:
        df = (
            df.sort_values(["day","count"], ascending=[True, False])
              .groupby("day")
              .head(15)
              .reset_index(drop=True)
        )

    # Merge mit Labels (und saubere Suffixe vermeiden)
    df = df.merge(labels, on="topic", how="left", validate="many_to_one")

    # Aufräumen/Sortieren
    sort_cols = ["day", "rank"] if "rank" in df.columns else ["day","count"]
    df = df.sort_values(sort_cols, ascending=[True, True])

    # Speichern
    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_CLEAN, index=False)

    # Sanity-Checks
    print(f"✅ saved: {OUT_CLEAN}")
    print("Max topics per day:", df.groupby("day")["topic"].count().max())
    if "rank" in df.columns:
        print("Max rank observed:", df.groupby("day")["rank"].max().max())
    print(df.head(50))

if __name__ == "__main__":
    main()
