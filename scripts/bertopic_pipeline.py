# we take the FinBERT approach from finbert_pipeline.py and adapt it to topics 
# that are mentioned in articles. 

 #!/usr/bin/env python3
import os
os.environ["USE_TF"] = "0"
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# -----------------------
# Grundeinstellungen
# -----------------------
FOLDERS = [
    Path("/Users/heiner/archive/2018_01_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_03_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_04_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_05_112b52537b67659ad3609a234388c50a"),
]
SNIPPET_PER_ARTICLE = 240     # kurzer Ausschnitt je Artikel (beschleunigt, reduziert Bias)
SAMPLE_MAX = 120_000          # max. Anzahl Artikel für das BERTopic-Fit (anpassbar)
TOP_K_PER_DAY = 15            # wie viele Top-Themen pro Tag speichern
OUT_TOPICS = Path("data/features/bertopic_daily_top15.parquet")
OUT_TOPIC_LABELS = Path("data/features/bertopic_topic_labels.parquet")
MODEL_DIR = Path("models/bertopic/")

# -----------------------
# Loader (wie gehabt)
# -----------------------
def extract_record(js: dict):
    dt = js.get("published") or js.get("thread", {}).get("published")
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    text = (js.get("text") or "").strip()
    if not text:
        return None
    return {
        "dt": dt,
        "url": js.get("url") or js.get("thread", {}).get("url"),
        "title": js.get("title") or js.get("thread", {}).get("title_full"),
        "publisher": js.get("thread", {}).get("site_full"),
        "text": text,
    }

def load_folder_recursive(root: Path) -> pd.DataFrame:
    rows = []
    files = list(root.rglob("*.json"))
    for p in files:
        try:
            with p.open("r", encoding="utf-8") as f:
                js = json.load(f)
            rec = extract_record(js)
            if rec:
                rows.append(rec)
        except Exception:
            continue
    return pd.DataFrame(rows)

# -----------------------
# Hauptprogramm
# -----------------------
if __name__ == "__main__":
    # 1) Daten laden
    parts = []
    for folder in FOLDERS:
        df_part = load_folder_recursive(folder)
        print(f"{folder} → {len(df_part)} articles")
        parts.append(df_part)
    df = pd.concat(parts, ignore_index=True)
    print("Total articles:", len(df))

    # 2) Day (NY) + Snippets
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"])
    df["dt_ny"] = df["dt"].dt.tz_convert(ZoneInfo("America/New_York"))
    df["day"] = df["dt_ny"].dt.floor("D")
    df["snippet"] = (
        df["text"].astype(str).str.slice(0, SNIPPET_PER_ARTICLE).str.replace(r"\s+", " ", regex=True)
    )

    # 3) Train/fit DOKUMENTE vorbereiten (Sampling für Geschwindigkeit)
    docs_all = df["snippet"].tolist()
    if len(docs_all) > SAMPLE_MAX:
        # gleichmäßiges Sample über die Zeit: per day n nehmen
        per_day = max(50, min(1000, SAMPLE_MAX // df["day"].nunique()))
        sampled_idx = (
            df.groupby("day")
              .head(per_day)
              .index
        )
        docs_fit = df.loc[sampled_idx, "snippet"].tolist()
        print(f"Fitting BERTopic on sampled docs: {len(docs_fit)} / {len(docs_all)}")
    else:
        docs_fit = docs_all
        print(f"Fitting BERTopic on all docs: {len(docs_fit)}")

    # 4) BERTopic initialisieren (leichtgewichtiges Embedding + simple Vectorizer)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectorizer_model = CountVectorizer(
        stop_words="english",
        max_df=0.9,          # sehr häufige Wörter kappen
        min_df=10,           # sehr seltene Wörter ignorieren
        ngram_range=(1, 2),  # etwas Kontext
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=100,      # zusammenhängendere Themen
        top_n_words=10,          # Labelqualität
        verbose=True,
        calculate_probabilities=False,
        low_memory=True,
    )

    # 5) Fit auf (gesampleten) Artikeln
    topics_fit, _ = topic_model.fit_transform(docs_fit)

    # 6) Alle Artikel transformieren (keine Re-Fits!)
    topics_all, _ = topic_model.transform(docs_all)
    df["topic"] = topics_all  # -1 == Outlier/Kein Thema

    # 7) Topic-Labels sichern (Mapping: topic_id -> Schlüsselwörter)
    topic_info = topic_model.get_topic_info()  # DataFrame von BERTopic
    # liefern: Topic, Count, Name (repräsentative Worte)
    topic_info.rename(columns={"Topic": "topic_id", "Name": "topic_label"}, inplace=True)
    OUT_TOPIC_LABELS.parent.mkdir(parents=True, exist_ok=True)
    topic_info[["topic_id", "topic_label", "Count"]].to_parquet(OUT_TOPIC_LABELS, index=False)
    print(f"Saved topic labels → {OUT_TOPIC_LABELS}")

    # 8) Pro Tag: Top-K Topics nach Häufigkeit
    # Outlier (-1) optional ausschließen:
    df_valid = df[df["topic"] != -1].copy()

    # Häufigkeit pro Tag & Topic
    counts = (
        df_valid.groupby(["day", "topic"])
                .size()
                .reset_index(name="count")
    )

    # Anteil pro Tag
    total_per_day = counts.groupby("day")["count"].sum().rename("day_total")
    counts = counts.merge(total_per_day, on="day", how="left")
    counts["share"] = counts["count"] / counts["day_total"]

    # Top-K pro Tag
    counts["rank"] = counts.groupby("day")["count"].rank(method="first", ascending=False)
    topk = counts[counts["rank"] <= TOP_K_PER_DAY].copy()

    # Labels mergen
    label_map = topic_info.set_index("topic_id")["topic_label"]
    topk["topic_label"] = topk["topic"].map(label_map)

    # Speichern
    OUT_TOPICS.parent.mkdir(parents=True, exist_ok=True)
    cols = ["day", "topic", "topic_label", "count", "share", "rank"]
    topk[cols].sort_values(["day", "rank"]).to_parquet(OUT_TOPICS, index=False)
    print(f"✅ Saved daily top-{TOP_K_PER_DAY} topics → {OUT_TOPICS}")
