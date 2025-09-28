import pandas as pd

ART = "/Users/heiner/stock-market-model/data/finbert/total_per_articles.parquet"     # per-article
DAY = "/Users/heiner/stock-market-model/data/finbert/total_per_day.parquet"        # per-day

art = pd.read_parquet(ART)
print("labels (per-article):")
print(art['sent_label'].value_counts(dropna=False))

# prüfe die signed-Scores
score_col = next((c for c in art.columns if c.lower() in ("sent_score_signed","ent_score_signed")), None)
print("score col:", score_col, art[score_col].describe())

# mittlere Längen – Hinweis auf zu lange Texte / Trunkierung
if "text" in art.columns:
    art["n_tokens_approx"] = art["text"].str.split().str.len()
    print(art["n_tokens_approx"].describe())

day = pd.read_parquet(DAY)
print("\nper-day columns:", list(day.columns)[:10], "…")
print(day[["day","mean_signed"]].head())
