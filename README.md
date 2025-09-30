# stock-market-model

1) stock-market-model/scripts/pipelines is used for data crawling and uses BERT models to process data. Also prices of the evaluation asset S&P 500 are being fetched here. 

2) stock-market-model/data stores the generated datasets from the pipelines and creates parquets or csv files to work with.

3) stock-market-model/out stores all the generated tables and figures used for visualization in the final report. 

To work with this repository, a JSON formatted dataset of any news articles can be processed. BERT models aim for text, title and other relevant news article which should be provided in a raw dataset.  
