from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# before the model does anything we need to load and prepare the raw data





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
