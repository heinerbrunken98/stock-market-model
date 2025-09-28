import re
import pandas as pd

path = "/Users/heiner/stock-market-model/data/bertopic/new/02_daily_top15.parquet"
tp = pd.read_parquet(path)

def collapse_earnings(lbl: str) -> str:
    if pd.isna(lbl): return lbl
    s = re.sub(r'^\s*\d+_', '', str(lbl))     # drop leading "35_"
    parts = [p.strip() for p in s.split('_') if p.strip()]
    out, seen_earn = [], False
    for p in parts:
        if 'earnings' in p.lower():
            if not seen_earn:
                out.append('earnings')
                seen_earn = True
        else:
            out.append(p)
    # shorten a bit
    out = [re.sub(r'\s+', ' ', x).strip() for x in out if x]
    return ' · '.join(out[:4])

tp["topic_label_pretty"] = tp["topic_label"].apply(collapse_earnings)

# Optional: per-topic eine einheitliche Bezeichnung wählen (modus)
canon = (tp.groupby("topic")["topic_label_pretty"]
           .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
tp["topic_label_pretty"] = tp["topic"].map(canon)

# (optional) Tag konvertieren, falls Unix ms
if pd.api.types.is_integer_dtype(tp["day"]):
    tp["day"] = pd.to_datetime(tp["day"], unit="ms")

# Ausgabe für deine Tabelle
cols = ["day","topic","topic_label_pretty","count","share","rank"]
tp[cols].sort_values(["day","rank"]).to_csv("out/02_daily_top15_pretty.csv", index=False)
print(f"Saved → {outpath.resolve()}")

