from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd


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

def load_folder(folder: Path) -> pd.DataFrame:
    # iterative over all json files in the folder
    rows = []
    for path in folder.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                js = json.load(f)
            rec = extract_record(js)
            if rec:
                rows.append(rec)
        except Exception:
            continue
    return pd.DataFrame(rows)






# FinBERT (tone) — Labelordnung:
# LABEL_0: neutral, LABEL_1: positive, LABEL_2: negative

# finbert tone seems to be the best model to use for this purpose -> Quelle 1 & 2
MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 64

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
        df_part = load_folder(folder)
        print(f"{folder} → {len(df_part)} articles")
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)
    print("Total combined:", len(df))
