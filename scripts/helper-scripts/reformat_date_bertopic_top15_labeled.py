from pathlib import Path
import pandas as pd

inp = Path("data/bertopic/bertopic_daily_top15_labeled.parquet")
out = Path("data/bertopic/bertopic_daily_top15_labeled_clean.parquet")

df = pd.read_parquet(inp)

# 1) 'day' nach NY-Zeit normalisieren (egal ob schon datetime oder ms-int)
if pd.api.types.is_integer_dtype(df["day"]):
    df["day"] = pd.to_datetime(df["day"], unit="ms", utc=True)
df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce").dt.tz_convert("America/New_York")

# 2) string-Datum erzeugen (überschreibt 'day' – optional)
df["day"] = df["day"].dt.strftime("%Y-%m-%d")

# 3) Optionale Aufräumarbeiten
df = df.rename(columns={"Count": "topic_global_count"})  # sprechender Name
df = df.sort_values(["day", "rank"])

out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print("✅ saved:", out)
