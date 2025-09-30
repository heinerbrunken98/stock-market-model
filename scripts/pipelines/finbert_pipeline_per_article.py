from pathlib import Path
from zoneinfo import ZoneInfo
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import json
from tqdm import tqdm
import argparse



def extract_record(js: dict) -> dict | None:
    dt = js.get("published") or js.get("thread", {}).get("published")
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return {
        "dt": dt,
        "url": js.get("thread", {}).get("url"),
        "title": js.get("thread", {}).get("title_full"),
        "publisher": js.get("thread", {}).get("site_full"),
        "text": js.get("text", "").strip(),
    }


def load_folder_recursive(root: Path) -> pd.DataFrame:
    rows = []
    files = list(root.rglob("*.json"))
    print(f"[scan] {root} -> {len(files)} files")
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                js = json.load(f)
            rec = extract_record(js)
            if rec:
                rows.append(rec)
        except Exception:
            continue
    return pd.DataFrame(rows)



MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 32
MAX_CHARS = 4000          # Länge je Artikel-Snippet (zur Sicherheit)
SNIPPET_PER_ARTICLE = 240 # wie gehabt
OUT_DIR = Path("data")


def build_pipeline():
    finbert = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 512
    return pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


def map_label(lbl: str) -> str:
    if lbl in ("LABEL_0", "Neutral"):  return "Neutral"
    if lbl in ("LABEL_1", "Positive"): return "Positive"
    if lbl in ("LABEL_2", "Negative"): return "Negative"
    return lbl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT pro Artikel – ein Ordner")
    parser.add_argument("folder", type=str, nargs="?", default=None,
                        help="Pfad zum News-Ordner (nur EIN Ordner wird verarbeitet)")
    parser.add_argument("--out-prefix", type=str, default="finbert",
                        help="Datei-Präfix im data/-Ordner (default: finbert)")
    args = parser.parse_args()

    # Ordner  bestimmen
    if args.folder is None:
        default_folder = Path("/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a")
        root = default_folder
        print(f"[info] Kein Ordner-Argument übergeben. Verwende Default: {root}")
    else:
        root = Path(args.folder)

    if not root.exists():
        raise SystemExit(f"Ordner nicht gefunden: {root}")

    # load data
    df = load_folder_recursive(root)
    if df.empty:
        raise SystemExit("Keine Artikel gefunden. Prüfe Pfad und Dateien.")

    print("Artikel gesamt:", len(df))

    # daytime
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    df["dt_ny"] = df["dt"].dt.tz_convert(ZoneInfo("America/New_York"))
    df["day"] = df["dt_ny"].dt.floor("D")

    # snippet 
    df["snippet"] = (
        df["text"].astype(str)
          .str.slice(0, SNIPPET_PER_ARTICLE)
          .str.replace(r"\s+", " ", regex=True)
          .str.slice(0, MAX_CHARS)
    )

    # finbert per article
    nlp = build_pipeline()
    labels, scores, signed = [], [], []

    texts = df["snippet"].tolist()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT per-article"):
        chunk = texts[i:i+BATCH_SIZE]
        out = nlp(chunk, batch_size=BATCH_SIZE, truncation=True, padding=True, max_length=512)
        for r in out:
            lbl = map_label(r["label"])
            sc  = float(r["score"])
            labels.append(lbl)
            scores.append(sc)
            signed.append(sc if lbl == "Positive" else (-sc if lbl == "Negative" else 0.0))

    df["sent_label"] = labels
    df["sent_score"] = scores
    df["sent_score_signed"] = signed

    # output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # per article results 
    out_articles = OUT_DIR / f"{args.out_prefix}_articles.parquet"
    df[[
        "dt", "dt_ny", "day", "publisher", "title", "url",
        "sent_label", "sent_score", "sent_score_signed"
    ]].to_parquet(out_articles, index=False)

    # daily aggregates 
    daily = (
        df.groupby("day").agg(
            n_articles=("sent_label", "size"),
            mean_signed=("sent_score_signed", "mean"),
            pos_share=("sent_label", lambda s: (s == "Positive").mean()),
            neg_share=("sent_label", lambda s: (s == "Negative").mean()),
        ).reset_index().sort_values("day")
    )

    # majority label per day
    def majority_label(s: pd.Series) -> str:
        counts = s.value_counts()
        if counts.empty:
            return "Neutral"
        top = counts.index[0]
        if counts.max() == 0:
            return "Neutral"
        if (counts == counts.max()).sum() > 1:
            return "Neutral"
        return top

    daily["majority_label"] = df.groupby("day")["sent_label"].apply(majority_label).values

    out_daily = "/Users/heiner/stock-market-model/data/finbert/02_per_article_finbert.parquet"
    daily.to_parquet(out_daily, index=False)

    print("✅ saved:")
    print("  • per article results  ", out_articles)
    print("  • daily aggregates (from articles)", out_daily)
