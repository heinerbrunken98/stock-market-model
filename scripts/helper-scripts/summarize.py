import pandas as pd

# Pfad zu deiner gespeicherten Datei
df = pd.read_parquet("data/daily_finbert.parquet")

# Alle Tage + Artikelanzahl anzeigen
print(df[["day", "n_articles"]])

# Summencheck: Wie viele Artikel insgesamt verarbeitet wurden
print("\nGesamtanzahl Artikel:", df["n_articles"].sum())
print("Anzahl Tage:", len(df))
