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

# finbert tone seems to be the best model to use for this purpose -> Quelle 1 & 2
MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 64
MAX_CHARS = 4000 

finbert = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)



# sentences = ["there is a shortage of capital, and we need extra financing",  
#             "growth is strong and we have plenty of liquidity", 
#             "there are doubts about our finances", 
#             "profits are flat"]
# results = nlp(sentences)
# print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative

if __name__ == "__main__":
    # news_article folders
    folders = [
        Path("/Users/heiner/archive/2018_01_112b52537b67659ad3609a234388c50a"),
        Path("/Users/heiner/archive/2018_02_112b52537b67659ad3609a234388c50a"),
        Path("/Users/heiner/archive/2018_03_112b52537b67659ad3609a234388c50a"),
        Path("/Users/heiner/archive/2018_04_112b52537b67659ad3609a234388c50a"),
        Path("/Users/heiner/archive/2018_05_112b52537b67659ad3609a234388c50a"),
    ]

    dfs = []
    for folder in folders:
        df_part = load_folder_recursive(folder)
        print(f"{folder} → {len(df_part)} articles")
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)
    print("Total combined:", len(df))

    if "text" not in df.columns:
        raise SystemExit("No 'text' column found after loading JSON. Check extract_record().")

    # remove empty or missing texts
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df.loc[df["text"].str.len() > 0].reset_index(drop=True)

    # create truncated text version for processing
    df["text_trunc"] = df["text"].str.slice(0, MAX_CHARS)
    
    # if df is empty after cleanting, exit
    if df.empty:
        raise SystemExit("After cleaning, no articles with non-empty text remained.")
    
    
    
    labels_raw = []
    labels_clean = []
    scores = [] 
    signed = []
    
    def map_label(lbl: str) -> str:
        if lbl in ("LABEL_0", "neutral"):  return "neutral"
        if lbl in ("LABEL_1", "positive"): return "positive"
        if lbl in ("LABEL_2", "negative"): return "negative"
        return lbl
    
    texts = df["text_trunc"].tolist()
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT"):
        out = nlp(texts[i:i+BATCH_SIZE], batch_size=BATCH_SIZE, truncation=True)

    for res in out:
        lbl_raw = res["label"]
        lbl = map_label(lbl_raw)
        sc = float(res["score"])
        labels_raw.append(lbl_raw)
        labels.append(lbl)
        scores.append(sc)
        signed.append(sc if lbl == "positive" else (-sc if lbl == "negative" else 0.0))

    df["sent_label_raw"] = labels_raw
    df["sent_label"] = labels
    df["sent_score"] = scores
    df["sent_score_signed"] = signed
    df["text_snippet"] = df["text"].str.slice(0, 200)
    
    # save results to parquet
    out_path = "data/processed/news_2018_01-05_finbert.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df[["dt","url","title","publisher","sent_label","sent_score","sent_score_signed","text_snippet"]].to_parquet(out_path, index=False)
    print(f"✅ saved {len(df)} rows → {out_path}")


