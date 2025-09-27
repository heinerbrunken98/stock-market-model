from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import json
from tqdm import tqdm


# before the model does anything we need to load and prepare the raw data

def extract_record(js: dict) -> dict | None:
    # transform date string to datetime object
    dt = js.get("published") or js.get("thread", {}).get("published")
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    
    # extract relevant fields
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
        except Exception as e:
            continue
    return pd.DataFrame(rows)


# FinBERT (tone) — Labelordnung:
# LABEL_0: neutral, LABEL_1: positive, LABEL_2: negative

FOLDERS = [
    Path("/Users/heiner/archive/2018_01_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_03_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_04_112b52537b67659ad3609a234388c50a"),
    Path("/Users/heiner/archive/2018_05_112b52537b67659ad3609a234388c50a"),
]

# finbert tone seems to be the best model to use for this purpose -> Quelle 1 & 2
MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 32
MAX_CHARS = 4000 

finbert = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

tokenizer.model_max_length = 512  # set max length for tokenizer
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# ==== FinBERT Setup ====
def build_pipeline():
    finbert = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 512
    return pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def map_label(lbl: str) -> str:
    if lbl in ("LABEL_0", "neutral"):  return "neutral"
    if lbl in ("LABEL_1", "positive"): return "positive"
    if lbl in ("LABEL_2", "negative"): return "negative"
    return lbl

if __name__ == "__main__":
    # load raw news articles
    parts = []
    for folder in FOLDERS:
        df_part = load_folder_recursive(folder)
        print(f"{folder} → {len(df_part)} articles")
        parts.append(df_part)
    if not parts or sum(len(p) for p in parts) == 0:
        raise SystemExit("No articles found. Check paths and file names.")
    df = pd.concat(parts, ignore_index=True)
    print("Total combined:", len(df))

    # build daily corpus (NY time, per article snippet)
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"])
    df["dt_ny"] = df["dt"].dt.tz_convert(ZoneInfo("America/New_York"))
    df["day"] = df["dt_ny"].dt.floor("D")

    # only snippet per day so that one long article does not dominate the sentiment
    df["snippet"] = df["text"].astype(str).str.slice(0, SNIPPET_PER_ARTICLE).str.replace(r"\s+", " ", regex=True)

    daily = (
        df.groupby("day")
          .agg(
              n_articles=("snippet", "size"),
              daily_text=("snippet", lambda s: " [SEP] ".join(s.tolist())[:200000])  # harte Kappung auf 200k chars
          )
          .reset_index()
          .sort_values("day")
    )
    print("Days:", len(daily))

    # FinBERT once per day
    nlp = build_pipeline()
    labels, scores, signed = [], [], []

    texts = daily["daily_text"].tolist()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT daily"):
        chunk = texts[i:i+BATCH_SIZE]
        out = nlp(chunk, batch_size=BATCH_SIZE, truncation=True, padding=True, max_length=512)
        for r in out:
            lbl = map_label(r["label"])
            sc  = float(r["score"])
            labels.append(lbl)
            scores.append(sc)
            signed.append(sc if lbl == "positive" else (-sc if lbl == "negative" else 0.0))

    daily["sent_label"] = labels
    daily["sent_score"] = scores
    daily["sent_score_signed"] = signed

    # save data output as parquet
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily[["day","n_articles","sent_label","sent_score","sent_score_signed"]].to_parquet(OUT_PATH, index=False)
    print(f"✅ saved: {OUT_PATH}  | days: {len(daily)}")


