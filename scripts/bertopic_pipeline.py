"""
Guided BERTopic — clearer macro topics per article (single folder)

This pipeline is tuned to surface crisp macro themes (e.g., earnings, inflation/CPI, Fed/Powell/rates,
trade/tariffs, jobs/NFP, oil) and to reduce PR/IR noise (appointments, generic dividends, boilerplate).

Key ideas
- Text prep: use title + body, normalize synonyms ("federal reserve"→"fed", etc.), add finance-specific stopwords,
  and allow up to trigrams so phrases like "consumer price index" survive.
- Clustering: smaller clusters (min_topic_size), explicit UMAP/HDBSCAN with a seed for reproducibility.
- Optional guidance: seed topics to gently pull documents into macro buckets.
- Representation: optionally use KeyBERT-inspired & Part-of-Speech representations for cleaner labels.

Outputs
- Topic labels: data/features/bertopic_topic_labels.parquet  (topic_id ↔ label words, counts)
- Daily Top-K:  data/features/bertopic_daily_top15.parquet  (per-day top topics with counts & shares)

CLI examples
- Default folder (Feb 2018):
    python bertopic_pipeline_guided_en.py
- Specific folder:
    python bertopic_pipeline_guided_en.py "/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a"
- With seed topics & multilingual embeddings:
    python bertopic_pipeline_guided_en.py --seeded --multilingual
- With post-hoc topic reduction to ~50 clusters (best-effort):
    python bertopic_pipeline_guided_en.py --seeded --reduce-topics 50
"""

import os
os.environ["USE_TF"] = "0"
import re
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import argparse

# Optional representations — guarded so the script still runs if the packages/models are missing
try:
    from bertopic.representation import KeyBERTInspired, PartOfSpeech
    HAVE_REP = True
except Exception:
    HAVE_REP = False

# Optional explicit clustering backends (recommended for reproducibility)
try:
    from umap import UMAP
    from hdbscan import HDBSCAN
    HAVE_CLUSTER = True
except Exception:
    HAVE_CLUSTER = False

# -----------------------
# Defaults / Paths
# -----------------------
SNIPPET_MAX_CHARS = 1200           # max characters for title+body per doc
SAMPLE_MAX = 120_000               # cap documents used for fitting (transform uses all)
TOP_K_PER_DAY = 15
OUT_TOPICS = Path("/Users/heiner/stock-market-model/data/bertopic/new/02_daily_top15.parquet")
OUT_TOPIC_LABELS = Path("/Users/heiner/stock-market-model/data/bertopic/new/02_topic_labels.parquet")

# -----------------------
# Loader
# -----------------------

def extract_record(js: dict):
    """Extract a minimal article record from a raw news JSON.
    Returns None if datetime is missing or both title and text are empty.
    """
    dt = js.get("published") or js.get("thread", {}).get("published")
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    text = (js.get("text") or "").strip()
    title = (js.get("title") or js.get("thread", {}).get("title_full") or "").strip()
    if not text and not title:
        return None
    return {
        "dt": dt,
        "url": js.get("url") or js.get("thread", {}).get("url"),
        "title": title,
        "publisher": js.get("thread", {}).get("site_full"),
        "text": text,
    }

def load_folder_recursive(root: Path) -> pd.DataFrame:
    """Recursively load all *.json files under a folder into a DataFrame of article records."""
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
            # swallow and continue; some files may be malformed
            continue
    return pd.DataFrame(rows)

# -----------------------
# Text normalization / domain rules
# -----------------------

# Finance/PR stopwords to de-emphasize boilerplate and corporate IR jargon
FINANCE_STOPS = {
    # PR/IR boilerplate, legal forms
    "inc","corp","co","company","ltd","plc","llc","press","release","announced","announces",
    "report","reports","reported","update","updates","today","news","businesswire","globenewswire",
    "prnewswire","nasdaq","nyse","marketwatch","reuters","bloomberg",
    # common market terms that wash out signal
    "share","shares","stock","stocks","market","markets","equity","equities","securities",
    "common","outstanding","board","executive","chief","officer","appointed","appointment",
    "dividend","quarter","quarterly","q1","q2","q3","q4","fiscal","guidance",
}

# Synonym/phrase normalization so key macro concepts cluster together
REPLACE_PATTERNS = [
    (r"\bfederal reserve\b", "fed"),
    (r"\bjerome powell\b|\bchair(wo)?man powell\b|\bpowell\b", "powell"),
    (r"\bfomc\b", "fed"),
    (r"\brate hike(s)?\b|\brate cut(s)?\b", "rates"),
    (r"\bconsumer price index\b|\bcpi\b", "cpi"),
    (r"\bproducer price index\b|\bppi\b", "ppi"),
    (r"\bnonfarm payrolls?\b|\bnfp\b|\bjobs report\b|\bunemployment\b", "jobs"),
    (r"\btariffs?\b|\btrade war\b|\bretaliation\b", "tariffs"),
    (r"\bbrent\b|\bwti\b|\bopec\b|\bcrude\b", "oil"),
    (r"\bearning(s)?\b|\bresults\b|\bprofit\b|\bnet income\b|\brevenue\b|\beps\b", "earnings"),
]

def normalize_text(s: str) -> str:
    """Lowercase + apply regex-based synonym normalization + whitespace squeeze."""
    s = (s or "").lower()
    for pat, rep in REPLACE_PATTERNS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided BERTopic — ONE folder, clearer macro topics per article")
    parser.add_argument("folder", type=str, nargs="?", default=None,
                        help="Path to the news folder (process ONE folder only)")
    parser.add_argument("--out-topics", type=str, default=str(OUT_TOPICS),
                        help="Output parquet for daily Top-K topics")
    parser.add_argument("--out-topic-labels", type=str, default=str(OUT_TOPIC_LABELS),
                        help="Output parquet for topic labels")
    parser.add_argument("--multilingual", action="store_true",
                        help="Use multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)")
    parser.add_argument("--seeded", action="store_true",
                        help="Enable seed topics (earnings, cpi, fed, tariffs, jobs, oil)")
    parser.add_argument("--reduce-topics", type=int, default=0,
                        help="Optionally reduce topics to N clusters after fitting (0=off)")
    parser.add_argument("--min-topic-size", type=int, default=50,
                        help="BERTopic min_topic_size / HDBSCAN min_cluster_size (default=50)")
    parser.add_argument("--max-df", type=float, default=0.90,
                        help="Vectorizer max_df (default=0.90)")
    parser.add_argument("--min-df", type=int, default=10,
                        help="Vectorizer min_df (default=10)")
    parser.add_argument("--ngram-max", type=int, default=3,
                        help="Use ngram_range=(1,N); default N=3")
    parser.add_argument("--doc-max-chars", type=int, default=SNIPPET_MAX_CHARS,
                        help="Max length for title+body doc; default=1200")
    parser.add_argument("--top-k-per-day", type=int, default=TOP_K_PER_DAY,
                        help="Keep Top-K topics per day; default=15")
    args = parser.parse_args()

    # Resolve folder (default: Feb 2018 folder)
    if args.folder is None:
        root = Path("/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a")
        print(f"[info] No folder argument given. Using default: {root}")
    else:
        root = Path(args.folder)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    # 1) Load data (ONE folder)
    df = load_folder_recursive(root)
    if df.empty:
        raise SystemExit("No articles found. Check path and files.")
    print("Total articles:", len(df))

    # 2) Build day index in NY time & construct per-article document (title + body)
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    df["dt_ny"] = df["dt"].dt.tz_convert(ZoneInfo("America/New_York"))
    df["day"] = df["dt_ny"].dt.floor("D")

    raw_doc = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.slice(0, args.doc_max_chars)
    df["doc"] = raw_doc.apply(normalize_text)

    # 3) Prepare fit documents (sample evenly over time if needed)
    docs_all = df["doc"].tolist()
    if len(docs_all) > SAMPLE_MAX:
        per_day = max(50, min(1000, SAMPLE_MAX // max(1, df["day"].nunique())))
        fit_idx = df.sort_values("dt").groupby("day").head(per_day).index
        docs_fit = df.loc[fit_idx, "doc"].tolist()
        print(f"Fitting BERTopic on sampled docs: {len(docs_fit)} / {len(docs_all)}")
    else:
        docs_fit = docs_all
        print(f"Fitting BERTopic on all docs: {len(docs_fit)}")

    # 4) Initialize models
    emb_name = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" if args.multilingual
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_model = SentenceTransformer(emb_name)

    # Build stopword list (English + finance/PR)
    base_stop = CountVectorizer(stop_words="english").get_stop_words()
    stop_words = list(set(base_stop).union(FINANCE_STOPS))

    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_range=(1, max(1, args.ngram_max)),
    )

    umap_model = None
    hdbscan_model = None
    if HAVE_CLUSTER:
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=max(5, args.min_topic_size), min_samples=5, metric="euclidean")

    # Optional seed topics to guide clustering toward macro themes
    seed_topic_list = None
    if args.seeded:
        seed_topic_list = [
            ["earnings","eps","revenue","guidance"],
            ["inflation","cpi","prices","wages"],
            ["fed","powell","rates","fomc"],
            ["tariffs","trade","china","retaliation"],
            ["jobs","employment","unemployment","nfp"],
            ["oil","crude","opec","brent","wti"],
        ]

    # Optional representation models for clearer labels
    rep_models = None
    if HAVE_REP:
        rep_models = [KeyBERTInspired(diversity=0.5)]
        try:
            rep_models.append(PartOfSpeech("en_core_web_sm", top_n=10, constraint="NOUN|PROPN"))
        except Exception:
            pass  # spaCy model not installed; KeyBERT-inspired alone still helps

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=args.min_topic_size,
        top_n_words=12,
        representation_model=rep_models,
        calculate_probabilities=False,
        low_memory=True,
        verbose=True,
        seed_topic_list=seed_topic_list,
    )

    # 5) Fit on (sampled) docs
    _topics_fit, _ = topic_model.fit_transform(docs_fit)

    # 6) Transform all documents (no refit)
    topics_all, _ = topic_model.transform(docs_all)
    df["topic"] = topics_all  # -1 = outlier/noise

    # 6b) Optional coarse reduction to N topics
    if args.reduce_topics and args.reduce_topics > 0:
        try:
            # Newer BERTopic returns (new_topics, new_probs) and updates the model in-place
            new_topics, _ = topic_model.reduce_topics(docs_all, topics=topics_all, nr_topics=args.reduce_topics)
            df["topic"] = new_topics
        except TypeError:
            try:
                # Some versions return a new model; try to capture it
                reduced_model, new_topics = topic_model.reduce_topics(docs_all, topics_all, nr_topics=args.reduce_topics)
                topic_model = reduced_model
                df["topic"] = new_topics
            except Exception as e:
                print(f"[warn] Topic reduction failed ({e}); continuing without reduction.")

    # 7) Persist topic labels (topic_id -> label words, counts)
    topic_info = topic_model.get_topic_info().rename(columns={"Topic": "topic_id", "Name": "topic_label"})
    OUT_TOPIC_LABELS = Path(args.out_topic_labels)
    OUT_TOPIC_LABELS.parent.mkdir(parents=True, exist_ok=True)
    topic_info[["topic_id", "topic_label", "Count"]].to_parquet(OUT_TOPIC_LABELS, index=False)
    print(f"Saved topic labels → {OUT_TOPIC_LABELS}")

    # 8) Per-day Top-K topics by frequency (exclude outliers)
    df_valid = df[df["topic"] != -1].copy()

    counts = (
        df_valid.groupby(["day", "topic"]).size().reset_index(name="count")
    )
    total_per_day = counts.groupby("day")["count"].sum().rename("day_total")
    counts = counts.merge(total_per_day, on="day", how="left")
    counts["share"] = counts["count"] / counts["day_total"]

    counts["rank"] = counts.groupby("day")["count"].rank(method="first", ascending=False)
    topk = counts[counts["rank"] <= args.top_k_per_day].copy()

    label_map = topic_info.set_index("topic_id")["topic_label"]
    topk["topic_label"] = topk["topic"].map(label_map)

    OUT_TOPICS = Path(args.out_topics)
    OUT_TOPICS.parent.mkdir(parents=True, exist_ok=True)
    cols = ["day", "topic", "topic_label", "count", "share", "rank"]
    topk[cols].sort_values(["day", "rank"]).to_parquet(OUT_TOPICS, index=False)

    print(f"✅ Saved daily top-{args.top_k_per_day} topics → {OUT_TOPICS}")
