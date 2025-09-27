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
