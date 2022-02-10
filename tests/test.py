import pandas as pd
from bunkatech.topic_modeling import topic_modeling

data = pd.read_csv("tests/data/imdb_movies.csv")

mask = data["description"].notna()
data = data[mask]
docs = data["description"].sample(1000).tolist()

res = topic_modeling(docs, topic_number="5")
print(res)
