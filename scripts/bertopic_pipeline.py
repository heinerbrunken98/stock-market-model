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
import argparse

# -----------------------
# Grundeinstellungen
# -----------------------
SNIPPET_PER_ARTICLE = 240     # kurzer Ausschnitt je Artikel (beschleunigt, reduziert Bias)
SAMPLE_MAX = 120_000          # max. Anzahl Artikel für das BERTopic-Fit (anpassbar)
TOP_K_PER_DAY = 15            # wie viele Top-Themen pro Tag speichern
OUT_TOPICS = Path("/Users/heiner/stock-market-model/data/bertopic/03_articles_top15_labeled.parquet")
OUT_TOPIC_LABELS = Path("/Users/heiner/stock-market-model/data/bertopic/03_articles_topic_labels.parquet")
MODEL_DIR = Path("models/bertopic/")

# -----------------------
# Loader
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
    print(f"[scan] {root} -> {len(files)} files")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTopic pro Artikel – EIN Ordner")
    parser.add_argument("folder", type=str, nargs="?", default=None,
                        help="Pfad zum News-Ordner (nur EIN Ordner wird verarbeitet)")
    parser.add_argument("--out-topics", type=str, default=str(OUT_TOPICS),
                        help="Pfad für daily Top-K Topics (parquet)")
    parser.add_argument("--out-topic-labels", type=str, default=str(OUT_TOPIC_LABELS),
                        help="Pfad für Topic-Labels (parquet)")
    parser.add_argument("--multilingual", action="store_true",
                        help="Multilinguales Embedding verwenden (paraphrase-multilingual-MiniLM-L12-v2)")
    args = parser.parse_args()

    # Ordner bestimmen (Default: 2018_02)
    if args.folder is None:
        root = Path("/Users/heiner/archive/2018_03_112b52537b67659ad3609a234388c50a")
        print(f"[info] Kein Ordner-Argument übergeben. Verwende Default: {root}")
    else:
        root = Path(args.folder)

    if not root.exists():
        raise SystemExit(f"Ordner nicht gefunden: {root}")

    # 1) Daten laden (nur EIN Ordner)
    df = load_folder_recursive(root)
    if df.empty:
        raise SystemExit("Keine Artikel gefunden. Prüfe Pfad und Dateien.")

    print("Total articles:", len(df))

    # 2) Day (NY) + Snippets
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df =df.dropna(subset=["dt"]) # defensive
df["dt_ny"] = df["dt"].dt.tz_convert(ZoneInfo("America/New_York"))
df["day"] = df["dt_ny"].dt.floor("D")
df["snippet"] = (
df["text"].astype(str)
.str.slice(0, SNIPPET_PER_ARTICLE)
.str.replace(r"\s+", " ", regex=True)
)


# 3) Train/fit DOKUMENTE vorbereiten (Sampling für Geschwindigkeit)
docs_all = df["snippet"].tolist()
if len(docs_all) > SAMPLE_MAX:
    per_day = max(50, min(1000, SAMPLE_MAX // max(1, df["day"].nunique())))
    sampled_idx = df.sort_values("dt").groupby("day").head(per_day).index
    docs_fit = df.loc[sampled_idx, "snippet"].tolist()
    print(f"Fitting BERTopic on sampled docs: {len(docs_fit)} / {len(docs_all)}")
else:
    docs_fit = docs_all
    print(f"Fitting BERTopic on all docs: {len(docs_fit)}")


# 4) BERTopic initialisieren (leichtgewichtiges Embedding + Vectorizer)
if args.multilingual:
    emb_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
else:
    emb_name = "sentence-transformers/all-MiniLM-L6-v2"


embedding_model = SentenceTransformer(emb_name)
vectorizer_model = CountVectorizer(
stop_words="english", # falls DE: mit --multilingual lassen wir Embeddings multilingual; Stopwörter nur im BoW
max_df=0.9,
min_df=10,
ngram_range=(1, 2),
)


topic_model = BERTopic(
embedding_model=embedding_model,
vectorizer_model=vectorizer_model,
min_topic_size=100,
top_n_words=10,
verbose=True,
calculate_probabilities=False,
low_memory=True,
)


# 5) Fit auf (gesampleten) Artikeln
topics_fit, _ = topic_model.fit_transform(docs_fit)


# 6) Alle Artikel transformieren (kein Re-Fit)
topics_all, _ = topic_model.transform(docs_all)
df["topic"] = topics_all # -1 == Outlier/Kein Thema


# 7) Topic-Labels sichern (Mapping: topic_id -> Schlüsselwörter)
topic_info = topic_model.get_topic_info().rename(columns={"Topic": "topic_id", "Name": "topic_label"})


OUT_TOPIC_LABELS = Path(args.out_topic_labels)
OUT_TOPIC_LABELS.parent.mkdir(parents=True, exist_ok=True)
topic_info[["topic_id", "topic_label", "Count"]].to_parquet(OUT_TOPIC_LABELS, index=False)
print(f"Saved topic labels → {OUT_TOPIC_LABELS}")


# 8) Pro Tag: Top-K Topics nach Häufigkeit (Outlier -1 ausschließen)
df_valid = df[df["topic"] != -1].copy()


counts = (
df_valid.groupby(["day", "topic"]).size().reset_index(name="count")
)
total_per_day = counts.groupby("day")["count"].sum().rename("day_total")
counts = counts.merge(total_per_day, on="day", how="left")
counts["share"] = counts["count"] / counts["day_total"]


counts["rank"] = counts.groupby("day")["count"].rank(method="first", ascending=False)
topk = counts[counts["rank"] <= TOP_K_PER_DAY].copy()


label_map = topic_info.set_index("topic_id")["topic_label"]
topk["topic_label"] = topk["topic"].map(label_map)


OUT_TOPICS = Path(args.out_topics)
OUT_TOPICS.parent.mkdir(parents=True, exist_ok=True)
cols = ["day", "topic", "topic_label", "count", "share", "rank"]
topk[cols].sort_values(["day", "rank"]).to_parquet(OUT_TOPICS, index=False)


print(f"✅ Saved daily top-{TOP_K_PER_DAY} topics → {OUT_TOPICS}")