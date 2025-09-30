from pathlib import Path
import pandas as pd


BASE = Path("/Users/heiner/stock-market-model")
PARQUET = BASE / "data/finbert/total_per_articles.parquet"
PUBLISHER_COL = "publisher"  

def main():
    df = pd.read_parquet(PARQUET)  
    if PUBLISHER_COL not in df.columns:
        raise KeyError(f"'{PUBLISHER_COL}' nicht in Spalten: {list(df.columns)}")

    counts = (
        df[PUBLISHER_COL]
        .astype(str).str.strip()
        .replace({"": pd.NA})
        .dropna()
        .value_counts()
        .rename_axis("publisher")
        .reset_index(name="count")
    )
    
    total = counts["count"].sum()


    print(counts.to_string(index=False))       
    print(f"Gesamtanzahl Artikel: {total}")
    

if __name__ == "__main__":
    main()
