#!/usr/bin/env python3
"""
Guided BERTopic — patched for clearer macro topics (single folder)

What changed vs. prior version
- Data cleanup: optional **PR-wire exclusion**, **title deduplication**, optional **macro-only** subset.
- Text shaping: synonym normalization + optional **macro keyword boosting**.
- Vectorizer: default **trigrams (1–3)**, stricter **max_df=0.85**, **min_df=20**.
- Clustering: UMAP(n_neighbors=30) + HDBSCAN(**prediction_data=True**, cluster_selection_method='leaf').
- Representation: KeyBERTInspired fallback across BERTopic versions; optional POS.
- Compatibility: robust to older BERTopic (no 'diversity' arg), optional seed topics if supported by your version.

Outputs (same as before)
- Topic labels parquet: data/features/bertopic_topic_labels.parquet
- Daily Top-K topics parquet: data/features/bertopic_daily_top15.parquet

Examples
    # default folder (Feb 2018), PR filtered, boost macro, macro-only OFF
    python bertopic_pipeline_guided_en_patched.py

    # explicit folder, macro-only ON, reduce to 50 topics post-hoc
    python bertopic_pipeline_guided_en_patched.py \
        "/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a" \
        --macro-only --reduce-topics 50

    # keep PR wires (not recommended), multilingual, seeds
    python bertopic_pipeline_guided_en_patched.py --include-pr --multilingual --seeded
"""

import os
os.environ["USE_TF"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import argparse

# Optional representations — guarded for compatibility
try:
    from bertopic.representation import KeyBERTInspired, PartOfSpeech
    HAVE_REP = True
except Exception:
    HAVE_REP = False

# Optional explicit clustering backends
try:
    from umap import UMAP
    from hdbscan import HDBSCAN
    HAVE_CLUSTER = True
except Exception:
    HAVE_CLUSTER = False

# -----------------------
# Defaults / Paths
# -----------------------
SNIPPET_MAX_CHARS = 1200
SAMPLE_MAX = 120_000
TOP_K_PER_DAY = 15
OUT_TOPICS = Path("/Users/heiner/stock-market-model/data/bertopic/new/03_daily_top15.parquet")
OUT_TOPIC_LABELS = Path("/Users/heiner/stock-market-model/data/bertopic/new/03_topic_labels.parquet")

PR_DOMAINS = ("businesswire", "globenewswire", "prnewswire")
MACRO_RE = re.compile(r"\b(fed|fomc|powell|rates?|cpi|inflation|ppi|tariffs?|trade|china|jobs|nfp|unemployment|oil|opec|wti|brent|crude|earnings|eps|revenue)\b")

# -----------------------
# Loader
# -----------------------

def extract_record(js: dict):
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

# -----------------------
# Text normalization and boosting
# -----------------------

FINANCE_STOPS = {
    # PR/IR boilerplate, legal forms
    "inc","corp","co","company","ltd","plc","llc","press","release","announced","announces",
    "report","reports","reported","update","updates","today","news","businesswire","globenewswire",
    "prnewswire","nasdaq","nyse","marketwatch","reuters","bloomberg",
    # generic market noise
    "share","shares","stock","stocks","market","markets","equity","equities","securities",
    "common","outstanding","board","executive","chief","officer","appointed","appointment",
    "dividend","quarter","quarterly","q1","q2","q3","q4","fiscal","guidance",
}

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

BOOST_RULES = [
    (r"\b(cpi|inflation|ppi)\b", " cpi inflation"),
    (r"\b(fed|fomc|powell|rates?)\b", " fed powell rates"),
    (r"\b(tariffs?|trade|china)\b", " tariffs trade"),
    (r"\b(jobs|nfp|unemployment)\b", " jobs nfp"),
    (r"\b(oil|opec|wti|brent|crude)\b", " oil"),
    (r"\b(earnings|eps|revenue)\b", " earnings"),
]

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    for pat, rep in REPLACE_PATTERNS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def boost_macro(s: str) -> str:
    out = s
    for pat, add in BOOST_RULES:
        if re.search(pat, out):
            out += add
    return out

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Guided BERTopic — clearer macro topics (single folder)")
    ap.add_argument("folder", type=str, nargs="?", default=None, help="Path to ONE news folder")
    ap.add_argument("--out-topics", type=str, default=str(OUT_TOPICS), help="Daily Top-K topics parquet path")
    ap.add_argument("--out-topic-labels", type=str, default=str(OUT_TOPIC_LABELS), help="Topic labels parquet path")
    ap.add_argument("--multilingual", action="store_true", help="Use paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--seeded", action="store_true", help="Try seed topics (if your BERTopic supports it)")
    ap.add_argument("--reduce-topics", type=int, default=0, help="Reduce topics to N after fitting (0=off)")
    ap.add_argument("--min-topic-size", type=int, default=40, help="HDBSCAN/BERTopic min_topic_size (default=40)")
    ap.add_argument("--max-df", type=float, default=0.85, help="Vectorizer max_df (default=0.85)")
    ap.add_argument("--min-df", type=int, default=20, help="Vectorizer min_df (default=20)")
    ap.add_argument("--ngram-max", type=int, default=3, help="Use ngram_range=(1,N); default N=3")
    ap.add_argument("--doc-max-chars", type=int, default=SNIPPET_MAX_CHARS, help="Doc length cap (title+body)")
    ap.add_argument("--top-k-per-day", type=int, default=TOP_K_PER_DAY, help="Keep Top-K topics per day")
    ap.add_argument("--include-pr", action="store_true", help="Keep PR wires (default: excluded)")
    ap.add_argument("--macro-only", action="store_true", help="Keep only documents matching macro regex")
    ap.add_argument("--no-boost", action="store_true", help="Disable macro keyword boosting")
    args = ap.parse_args()

    # Resolve folder (default: Feb 2018)
    if args.folder is None:
        root = Path("/Users/heiner/archive/2018_03_112b52537b67659ad3609a234388c50a")
        print(f"[info] No folder argument given. Using default: {root}")
    else:
        root = Path(args.folder)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    # Load
    df = load_folder_recursive(root)
    if df.empty:
        raise SystemExit("No articles found. Check path and files.")
    print("Total raw articles:", len(df))

    # Time index (NY) and document build
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    df["dt_ny"] = df["dt"].dt.tz_convert(ZoneInfo("America/New_York"))
    df["day"] = df["dt_ny"].dt.floor("D")

    # PR filter (default ON)
    if not args.include_pr:
        mask_pr = df["publisher"].fillna("").str.contains("|".join(PR_DOMAINS), case=False)
        before = len(df)
        df = df.loc[~mask_pr].copy()
        print(f"[filter] Excluded PR wires: {before - len(df)} removed → {len(df)} remain")

    # Build doc: title + body → normalize → optional macro-only → dedup titles → optional boosting
    raw_doc = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.slice(0, args.doc_max_chars)
    df["doc"] = raw_doc.apply(normalize_text)

    if args.macro_only:
        before = len(df)
        df = df.loc[df["doc"].str.contains(MACRO_RE)].copy()
        print(f"[filter] Macro-only: {before - len(df)} removed → {len(df)} remain")

    # Title dedup
    df["title_norm"] = df["title"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    before = len(df)
    df = df.drop_duplicates(subset=["title_norm"]).copy()
    print(f"[dedup] Title dedup: {before - len(df)} removed → {len(df)} remain")

    # Optional boosting
    if not args.no_boost:
        df["doc"] = df["doc"].apply(boost_macro)

    # Prepare fit docs (balanced over days if large)
    docs_all = df["doc"].tolist()
    if len(docs_all) > SAMPLE_MAX:
        per_day = max(50, min(1000, SAMPLE_MAX // max(1, df["day"].nunique())))
        fit_idx = df.sort_values("dt").groupby("day").head(per_day).index
        docs_fit = df.loc[fit_idx, "doc"].tolist()
        print(f"Fitting BERTopic on sampled docs: {len(docs_fit)} / {len(docs_all)}")
    else:
        docs_fit = docs_all
        print(f"Fitting BERTopic on all docs: {len(docs_fit)}")

    # Initialize models
    emb_name = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" if args.multilingual
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_model = SentenceTransformer(emb_name)

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
        umap_model = UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(
            min_cluster_size=max(5, args.min_topic_size),
            min_samples=5,
            metric="euclidean",
            prediction_data=True,
            cluster_selection_method="leaf",
        )

    # Seed topics (best-effort, may be ignored on older versions)
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

    # Representation models (fallback across versions)
    rep_models = None
    if HAVE_REP:
        try:
            rep = KeyBERTInspired(diversity=0.5)
        except TypeError:
            rep = KeyBERTInspired()
        rep_models = [rep]
        try:
            rep_models.append(PartOfSpeech("en_core_web_sm", top_n=10, constraint="NOUN|PROPN"))
        except Exception:
            pass

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

    # Fit on (sampled) docs
    _topics_fit, _ = topic_model.fit_transform(docs_fit)

    # Transform all docs
    topics_all, _ = topic_model.transform(docs_all)
    df["topic"] = topics_all  # -1 = outlier/noise

    # Optional reduce
    if args.reduce_topics and args.reduce_topics > 0:
        try:
            new_topics, _ = topic_model.reduce_topics(docs_all, topics=topics_all, nr_topics=args.reduce_topics)
            df["topic"] = new_topics
        except TypeError:
            try:
                reduced_model, new_topics = topic_model.reduce_topics(docs_all, topics_all, nr_topics=args.reduce_topics)
                topic_model = reduced_model
                df["topic"] = new_topics
            except Exception as e:
                print(f"[warn] Topic reduction failed ({e}); continuing without reduction.")

    # Save labels
    topic_info = topic_model.get_topic_info().rename(columns={"Topic": "topic_id", "Name": "topic_label"})
    OUT_TOPIC_LABELS = Path(args.out_topic_labels)
    OUT_TOPIC_LABELS.parent.mkdir(parents=True, exist_ok=True)
    topic_info[["topic_id", "topic_label", "Count"]].to_parquet(OUT_TOPIC_LABELS, index=False)
    print(f"Saved topic labels → {OUT_TOPIC_LABELS}")

    # Daily Top-K (exclude -1)
    df_valid = df[df["topic"] != -1].copy()
    counts = df_valid.groupby(["day", "topic"]).size().reset_index(name="count")
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
