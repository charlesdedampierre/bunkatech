from bunkatech.topic_modeling import topic_modeling
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
df = pd.DataFrame(docs)
sample_size = 2000
df = df.sample(sample_size).reset_index(drop=True)
df.columns = ["text"]
df["bindex"] = df.index

#df.to_csv("sample_test.csv", index=False)
#df = pd.read_csv("sample_test.csv")


model = topic_modeling()
bunka = model.fit(
    df,
    index_var="bindex",
    text_var="text",
    sample_size=2000,
    max_terms=100,
    topic_number=20,
    top_terms=3,
    ngrams=(1, 2),
    ents=False,
    sample_terms=2000,
    model="tfidf",
    language="en",
)
fig = model.visualize_embeddings()
fig.show()
